"""Native titlebar styling helpers for the Qt shell."""

from __future__ import annotations

import ctypes
import sys


def _normalize_theme(theme: str | None = "dark") -> str:
    value = str(theme or "dark").strip().lower()
    return "light" if value == "light" else "dark"


def _theme_tokens(theme: str = "dark") -> dict[str, str | bool]:
    if _normalize_theme(theme) == "light":
        return {
            "dark": False,
            "bg": "#f6f7f9",
            "surface": "#ffffff",
            "surface_2": "#eef2f6",
            "text": "#1f1f1f",
            "accent": "#4a90e2",
            "border": "#d8dde5",
        }
    return {
        "dark": True,
        "bg": "#141414",
        "surface": "#1e1e1e",
        "surface_2": "#282828",
        "text": "#d8d8d8",
        "accent": "#4a90e2",
        "border": "#555555",
    }


def configure_native_titlebar(
    window,
    theme: str = "dark",
    *,
    expand_macos: bool = False,
    update_qt_flags: bool = True,
) -> str:
    """Blend the OS titlebar with the active FiLark theme where possible."""
    tokens = _theme_tokens(theme)
    _apply_window_palette(window, tokens)

    style = "native"
    if sys.platform == "darwin":
        if expand_macos and update_qt_flags:
            _apply_macos_expanded_flags(window)
        if _apply_macos_titlebar(window, tokens):
            style = "macos-cocoa"
        elif expand_macos:
            style = "macos-transparent"
    elif sys.platform.startswith("win"):
        if _apply_windows_titlebar(window, tokens):
            style = "windows-dwm"

    window.setProperty("_filark_native_titlebar", style)
    return style


def perform_native_titlebar_double_click(window) -> bool:
    """Run the platform titlebar double-click action."""
    if sys.platform == "darwin":
        if _perform_macos_titlebar_double_click(window):
            return True
    try:
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()
        return True
    except Exception:
        return False


def uses_macos_custom_titlebar(style: str | None) -> bool:
    """Return whether this chrome mode needs a macOS hit area."""
    return sys.platform == "darwin" and str(style or "").startswith("macos-")


def _apply_window_palette(window, tokens: dict[str, str | bool]) -> None:
    try:
        from PySide6.QtGui import QColor, QPalette
    except Exception:
        return

    palette = window.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(str(tokens["bg"])))
    palette.setColor(QPalette.ColorRole.Base, QColor(str(tokens["surface"])))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(str(tokens["surface_2"])))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(str(tokens["text"])))
    palette.setColor(QPalette.ColorRole.Text, QColor(str(tokens["text"])))
    palette.setColor(QPalette.ColorRole.Button, QColor(str(tokens["surface_2"])))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(str(tokens["text"])))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(str(tokens["accent"])))
    window.setPalette(palette)


def _apply_macos_expanded_flags(window) -> None:
    try:
        from PySide6.QtCore import Qt
    except Exception:
        return

    expanded = getattr(Qt.WindowType, "ExpandedClientAreaHint", None)
    no_titlebar_bg = getattr(Qt.WindowType, "NoTitleBarBackgroundHint", None)
    if expanded is not None and no_titlebar_bg is not None:
        window.setWindowFlag(expanded, True)
        window.setWindowFlag(no_titlebar_bg, True)
    if hasattr(window, "setUnifiedTitleAndToolBarOnMac"):
        window.setUnifiedTitleAndToolBarOnMac(True)


def _apply_macos_titlebar(window, tokens: dict[str, str | bool]) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None and app.platformName().lower() != "cocoa":
            return False

        objc = _objc_runtime()
        view = int(window.winId())
        ns_window = _ns_window_for_view(objc, view)
        if not ns_window:
            return False

        style_mask = _objc_msg(objc, ns_window, "styleMask", ctypes.c_ulong)
        style_mask |= 1 << 15  # NSWindowStyleMaskFullSizeContentView
        _objc_msg(objc, ns_window, "setStyleMask:", None, ctypes.c_ulong, style_mask)
        _objc_msg(objc, ns_window, "setTitlebarAppearsTransparent:", None, ctypes.c_bool, True)
        _objc_msg(objc, ns_window, "setTitleVisibility:", None, ctypes.c_long, 1)
        _objc_msg(objc, ns_window, "setMovableByWindowBackground:", None, ctypes.c_bool, False)
        _objc_msg(
            objc,
            ns_window,
            "setBackgroundColor:",
            None,
            ctypes.c_void_p,
            _ns_color(objc, str(tokens["bg"])),
        )

        appearance_name = "NSAppearanceNameDarkAqua" if bool(tokens["dark"]) else "NSAppearanceNameAqua"
        appearance = _ns_appearance(objc, appearance_name)
        if appearance:
            ns_app = _objc_msg(
                objc,
                objc.objc_getClass(b"NSApplication"),
                "sharedApplication",
                ctypes.c_void_p,
            )
            if ns_app:
                _objc_msg(objc, ns_app, "setAppearance:", None, ctypes.c_void_p, appearance)
            _objc_msg(objc, ns_window, "setAppearance:", None, ctypes.c_void_p, appearance)

        if _objc_responds_to(objc, ns_window, "setTitlebarSeparatorStyle:"):
            _objc_msg(objc, ns_window, "setTitlebarSeparatorStyle:", None, ctypes.c_long, 0)
        _paint_macos_frame_view(objc, ns_window, str(tokens["bg"]))
        return True
    except Exception:
        return False


def _perform_macos_titlebar_double_click(window) -> bool:
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None and app.platformName().lower() != "cocoa":
            return False

        objc = _objc_runtime()
        view = int(window.winId())
        ns_window = _ns_window_for_view(objc, view)
        if not ns_window:
            return False

        action = _macos_titlebar_double_click_action(objc)
        if action == "None":
            return True
        selector = "performMiniaturize:" if action == "Minimize" else "performZoom:"
        _objc_msg(objc, ns_window, selector, None, ctypes.c_void_p, 0)
        return True
    except Exception:
        return False


def _macos_titlebar_double_click_action(objc) -> str:
    defaults = objc.objc_getClass(b"NSUserDefaults")
    if not defaults:
        return "Maximize"
    standard_defaults = _objc_msg(objc, defaults, "standardUserDefaults", ctypes.c_void_p)
    if not standard_defaults:
        return "Maximize"
    key = _ns_string(objc, "AppleActionOnDoubleClick")
    action = _objc_msg(
        objc,
        standard_defaults,
        "stringForKey:",
        ctypes.c_void_p,
        ctypes.c_void_p,
        key,
    )
    if not action:
        return "Maximize"
    return _ns_string_value(objc, int(action)) or "Maximize"


def _apply_windows_titlebar(window, tokens: dict[str, str | bool]) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        hwnd = int(window.winId())
        dwmapi = ctypes.windll.dwmapi
        enabled = ctypes.c_int(1 if bool(tokens["dark"]) else 0)
        for attribute in (20, 19):
            dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(attribute),
                ctypes.byref(enabled),
                ctypes.sizeof(enabled),
            )
        for attribute, color in (
            (34, tokens["border"]),
            (35, tokens["bg"]),
            (36, tokens["text"]),
        ):
            colorref = ctypes.c_int(_windows_colorref(str(color)))
            dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(hwnd),
                ctypes.c_uint(attribute),
                ctypes.byref(colorref),
                ctypes.sizeof(colorref),
            )
        return True
    except Exception:
        return False


def _objc_runtime():
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    return objc


def _objc_selector(objc, selector: str):
    return ctypes.c_void_p(objc.sel_registerName(selector.encode("utf-8")))


def _objc_msg(objc, receiver: int, selector: str, restype, *args):
    argtypes = []
    values = []
    for index in range(0, len(args), 2):
        argtypes.append(args[index])
        values.append(args[index + 1])
    objc.objc_msgSend.restype = ctypes.c_void_p if restype is None else restype
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    result = objc.objc_msgSend(ctypes.c_void_p(receiver), _objc_selector(objc, selector), *values)
    return None if restype is None else result


def _objc_responds_to(objc, receiver: int, selector: str) -> bool:
    return bool(
        _objc_msg(
            objc,
            receiver,
            "respondsToSelector:",
            ctypes.c_bool,
            ctypes.c_void_p,
            _objc_selector(objc, selector),
        )
    )


def _ns_window_for_view(objc, view: int) -> int:
    if _objc_responds_to(objc, view, "window"):
        ns_window = _objc_msg(objc, view, "window", ctypes.c_void_p)
        if ns_window:
            return int(ns_window)
    return int(view)


def _paint_macos_frame_view(objc, ns_window: int, color: str) -> None:
    content_view = _objc_msg(objc, ns_window, "contentView", ctypes.c_void_p)
    if not content_view or not _objc_responds_to(objc, int(content_view), "superview"):
        return
    frame_view = _objc_msg(objc, int(content_view), "superview", ctypes.c_void_p)
    if not frame_view:
        return
    _objc_msg(objc, int(frame_view), "setWantsLayer:", None, ctypes.c_bool, True)
    layer = _objc_msg(objc, int(frame_view), "layer", ctypes.c_void_p)
    if not layer:
        return
    ns_color = _ns_color(objc, color)
    if not ns_color or not _objc_responds_to(objc, ns_color, "CGColor"):
        return
    cg_color = _objc_msg(objc, ns_color, "CGColor", ctypes.c_void_p)
    if cg_color:
        _objc_msg(objc, int(layer), "setBackgroundColor:", None, ctypes.c_void_p, cg_color)


def _ns_color(objc, value: str) -> int:
    red, green, blue = _rgb_floats(value)
    ns_color = objc.objc_getClass(b"NSColor")
    return int(
        _objc_msg(
            objc,
            ns_color,
            "colorWithCalibratedRed:green:blue:alpha:",
            ctypes.c_void_p,
            ctypes.c_double,
            red,
            ctypes.c_double,
            green,
            ctypes.c_double,
            blue,
            ctypes.c_double,
            1.0,
        )
    )


def _ns_appearance(objc, name: str) -> int:
    ns_appearance = objc.objc_getClass(b"NSAppearance")
    ns_name = _ns_string(objc, name)
    if not ns_appearance or not ns_name:
        return 0
    return int(
        _objc_msg(
            objc,
            ns_appearance,
            "appearanceNamed:",
            ctypes.c_void_p,
            ctypes.c_void_p,
            ns_name,
        )
        or 0
    )


def _ns_string(objc, value: str) -> int:
    ns_string = objc.objc_getClass(b"NSString")
    if not ns_string:
        return 0
    return int(
        _objc_msg(
            objc,
            ns_string,
            "stringWithUTF8String:",
            ctypes.c_void_p,
            ctypes.c_char_p,
            value.encode("utf-8"),
        )
        or 0
    )


def _ns_string_value(objc, ns_string: int) -> str:
    value = _objc_msg(objc, ns_string, "UTF8String", ctypes.c_void_p)
    if not value:
        return ""
    raw = ctypes.cast(ctypes.c_void_p(value), ctypes.c_char_p).value
    return raw.decode("utf-8") if raw else ""


def _rgb_floats(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _windows_colorref(value: str) -> int:
    value = value.lstrip("#")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return red | (green << 8) | (blue << 16)
