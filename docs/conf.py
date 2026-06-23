from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "torch-einshard"
author = "torch-einshard contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
]

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "torch-einshard"

autodoc_typehints = "description"
nitpicky = False
