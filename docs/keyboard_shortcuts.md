# Keyboard Shortcuts

Every command below is also listed in the menu bar (File, View, Analyze,
Help), which shows its current shortcut. Most shortcuts require a loaded source. Click the canvas first if the side panel
currently owns keyboard focus.

## Application

| Shortcut | Action |
|---|---|
| `Esc` | Toggle the side panel. |
| Windows/Linux `Ctrl+G`, macOS `Cmd+G` | Go to sample/time/UTC position. |
| Windows/Linux `Ctrl+O`, macOS `Cmd+O` | Open a file. |
| Windows/Linux `Ctrl+Alt+O`, macOS `Cmd+Option+O` | Open a folder. |
| Windows/Linux `Ctrl+W`, macOS `Cmd+W` | Close the current source. |
| Windows/Linux `Ctrl+I`, macOS `Cmd+I` | Source properties (metadata and overrides). |
| Windows/Linux `Ctrl+Shift+O`, macOS `Cmd+Shift+O` | Navigator: whole-file image of the data under the current view; click to jump. |
| `Alt+T` / `Alt+Shift+T` / `Alt+0` | Next / previous / raw tape transform. |
| Windows/Linux `Ctrl+Left`, macOS `Cmd+Left` | Previous comparison source. |
| Windows/Linux `Ctrl+Right`, macOS `Cmd+Right` | Next comparison source. |

## Navigation

| Shortcut | Action |
|---|---|
| `Left` / `Right` | Pan time. |
| `Up` / `Down` | Pan channels. |
| `Page Up` / `Page Down` | Page channels. |
| `Alt+Up` / `Alt+Down` | Alternative page channel controls. |

## Mouse And Trackpad

| Gesture | Action |
|---|---|
| Vertical scroll / pinch | Zoom the time axis. |
| `Shift` + scroll | Pan channels. |
| Horizontal scroll / drag | Pan time. |

## Display Scale

| Shortcut | Action |
|---|---|
| `X` | Increase time compression. |
| `Shift+X` | Decrease time compression. |
| `Y` | Increase channel compression. |
| `Shift+Y` | Decrease channel compression. |
| `F` | Fit channels into the Y buffer. |
| `Shift+F` | Reset Y compression. |
| `A` | Reduce display aspect. |
| `Shift+A` | Increase display aspect. |

## Realtime Playback

| Shortcut | Action |
|---|---|
| `R` | Toggle realtime auto-scroll. |
| `S` | Stop realtime auto-scroll. |
| `+` / `=` | Speed up playback. |
| `-` / `_` | Slow down playback. |
| `1`..`8` | Select configured realtime sample-rate presets. |

## Diagnostics

| Shortcut | Action |
|---|---|
| `P` | Print realtime debug stats. |
| `Shift+P` | Print showcase timing and memory stats. |
| `B` | Run a short throughput benchmark. |
| `M` | Print memory snapshot history when available. |

## Annotation

| Shortcut / gesture | Action |
|---|---|
| `Ctrl+drag` in BBox mode | Draw bbox. |
| `Ctrl+click`, then click in Polyline mode | Build polyline. |
| `Enter` / `Right-click` | Finish current polyline. |
| `Esc` | Cancel current draw/edit session. |
| Drag selected bbox handle | Resize bbox. |
| Drag selected polyline vertex | Move vertex. |
| `Shift+click` polyline segment | Insert vertex. |
| `Alt/Option+click` polyline vertex | Delete vertex. |
| `Ctrl+drag` in Mask Brush mode | Paint selected/new mask. |
| `Shift` while mask painting | Erase. |
| `Ctrl+Z` / `Ctrl+Y` | Mask brush undo/redo. |

On macOS, annotation drawing uses the Control key for canvas gestures. `Cmd+G`
is only the application-level Go To shortcut.
