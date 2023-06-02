# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyTensorRT'
copyright = '2023, amirHossein Malekzadeh'
author = 'amirHossein Malekzadeh'
release = '0.1'

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',     #Auto doc from docstring
               'sphinx.ext.napoleon',   #recognize numpy and google docstring
               'sphinx_copybutton',     #copy button for code block
               'sphinx_tabs.tabs',           #tabs for sphinix
              ] 

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
