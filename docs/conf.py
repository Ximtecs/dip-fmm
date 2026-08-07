"""Sphinx configuration for the dip-fmm documentation."""

from pathlib import Path
import subprocess


DOCS_DIR = Path(__file__).resolve().parent
DOXYGEN_BUILD_DIR = DOCS_DIR / "_build" / "doxygen"

# Generate XML on every clean documentation build; generated API data remains
# under _build and is intentionally excluded from version control.
# Doxygen requires its nested output directory to exist before it starts.
DOXYGEN_BUILD_DIR.mkdir(parents=True, exist_ok=True)
subprocess.run(["doxygen", "Doxyfile"], cwd=DOCS_DIR, check=True)

project = "dip-fmm"
copyright = "2026, dip-fmm contributors"

extensions = ["myst_parser", "breathe"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

breathe_projects = {"dip-fmm": str(DOCS_DIR / "_build/doxygen/xml")}
breathe_default_project = "dip-fmm"

html_theme = "sphinx_rtd_theme"
html_title = "dip-fmm documentation"

myst_enable_extensions = ["amsmath", "dollarmath"]
