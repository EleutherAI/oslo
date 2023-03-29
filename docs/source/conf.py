import os
import sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))
project = "OSLO"
copyright = "2023, EleutherAI"
author = "EleutherAI"
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
imgmath_image_format = 'png'
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
imgmath_embed = True
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"
language = None
exclude_patterns = []
pygments_style = "sphinx"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
latex_elements = {}
latex_documents = [
    (
        master_doc,
        "OSLO.tex",
        "OSLO Documentation",
        "EleutherAI",
        "manual",
    ),
]
man_pages = [
    (master_doc, "oslo", "OSLO Documentation", [author], 1)
]
texinfo_documents = [
    (
        master_doc,
        "OSLO",
        "OSLO Documentation",
        author,
        "OSLO",
        "One line description of project.",
        "Miscellaneous",
    ),
]
html_sidebars = {
'CONFIGURATION': [
                 'localtoc.html',
                 'relations.html',
                 'searchbox.html',
                 'foo.html',
            ]

        }
epub_title = project
epub_exclude_files = ["search.html"]
intersphinx_mapping = {"https://docs.python.org/": None}
todo_include_todos = True
