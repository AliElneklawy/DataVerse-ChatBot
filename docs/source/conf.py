import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

# Project information
project = 'DataVerse Chatbot'
copyright = '2023-2025, DataVerse Team'
author = 'DataVerse Team'
release = '2.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_inherit_docstrings = True
autoclass_content = 'both'

# Add modules that might not be available during docs build
autodoc_mock_imports = [
    'anthropic', 'arabic_reshaper', 'beautifulsoup4', 'cchardet', 'cohere',
    'crawl4ai', 'dnspython', 'docling', 'docx2txt', 'faiss_cpu', 'fastapi',
    'flask', 'google_generativeai', 'html2text', 'joblib', 'langchain',
    'langchain_community', 'matplotlib', 'mistralai', 'numpy', 'openai',
    'pandas', 'playsound', 'protobuf', 'pyaudio', 'pymupdf', 'requests',
    'tenacity', 'werkzeug', 'uvicorn', 'xgboost', 'pydantic'
]