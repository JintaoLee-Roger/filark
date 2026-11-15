## run: python -m http.server 8000 --bind localhost -d docs/_build/html
"""Sphinx configuration for the FiLark documentation site."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as get_version

project = "FiLark"
author = "Jintao Li"
copyright = "2026, Jintao Li"

try:
    release = get_version("filark")
except PackageNotFoundError:
    release = "0.0.0"

version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "markdown",
}
master_doc = "index"

myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

html_theme = "furo"
html_title = "FiLark"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "custom_sidebar_navigation.jinja",
        "sidebar/scroll-end.html",
    ],
}

html_theme_options = {
    "source_repository": "https://github.com/JintaoLee-Roger/filark/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_button": "edit",
    "light_css_variables": {
        "color-brand-primary": "#1767d5",
        "color-brand-content": "#1767d5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6ea8ff",
        "color-brand-content": "#6ea8ff",
    },
}
