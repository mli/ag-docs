
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

# project = "Practical Machine Learning"
copyright = "2021, All authors. Licensed under CC-BY-SA-4.0 and MIT-0."
author = "Alex Smola, Qingqing Huang, Mu Li"

extensions = [
    'myst_parser', 
    'sphinx_design', 
    'sphinx_copybutton',
    # 'nbsphinx',
    'sphinx_togglebutton',
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode"
    ]#"sphinxcontrib.bibtex","sphinxcontrib.rsvgconverter","sphinx.ext.autodoc","sphinx.ext.viewcode"]
myst_enable_extensions = ["colon_fence", "deflist", "substitution", "html_image"]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md', 'syllabus_raw.md']
master_doc = 'index'
numfig = True
numfig_secnum_depth = 2
math_numfig = True
math_number_all = True

suppress_warnings = ['misc.highlighting_failure']

# html_title = project
# html_theme = 'sphinx_material'
html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "autogluon.png",
    "dark_logo": "autogluon-w.png",

    # 'base_url': 'http://bashtage.github.io/sphinx-material/',
    'repo_url': 'https://github.com/awslabs/autogluon/',    
    'repo_name': 'AutoGluon',
    # 'google_analytics_account': 'UA-96378503-12',
    # 'html_minify': True,
    # 'css_minify': True,
    #'nav_title': 'Practical Machine Learning',
    # 'logo_icon': '&#xe869',
    'globaltoc_depth': 2,
    "color_primary": "blue",
    # 'navigation_depth': 4,
    'globaltoc_collapse': False,
    'master_doc': False,
#    'nav_links': [{'href':'index', 'title':'Home', 'internal':True}],
    "light_css_variables": {
        "color-brand-primary": "#3977B9",
        "color-brand-content": "#3977B9",
    },
    "announcement": "Check new release 0.5 with forcast and multi-modal!",
    
}


# html_sidebars = {
#     "**": [#"logo-text.html", 
#     "globaltoc.html", "localtoc.html", "searchbox.html"]
# }

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

html_static_path = ['_static']

html_favicon = '_static/favicon.ico'

# html_logo = '_static/autogluon.png'

html_css_files = [
    'custom.css', 'https://at.alicdn.com/t/font_2371118_b27k2sys2hd.css'
]

def setup(app):
    pass
