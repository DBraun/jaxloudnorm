"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Project information
project = 'jaxloudnorm'
copyright = '2025, David Braun, Boris Kuznetsov, Christian Steinmetz'
author = 'David Braun, Boris Kuznetsov, Christian Steinmetz'

# Get version from package
import jaxloudnorm
release = jaxloudnorm.__version__
version = release

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
add_module_names = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}