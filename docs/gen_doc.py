# docs/gen_doc.py
import os
import sys
import subprocess
import shutil

# Add the project root to the Python path so imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Change to docs directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories
os.makedirs("source/_static", exist_ok=True)
os.makedirs("source/_templates", exist_ok=True)

# Create custom CSS file for larger font sizes
css_content = """
body {
    font-size: 16px;  /* Increase base font size */
}

.body {
    font-size: 16px;  /* Increase content font size */
}

pre, code {
    font-size: 14px;  /* Increase code font size */
}

h1 {
    font-size: 32px;  /* Increase heading font size */
}

h2 {
    font-size: 24px;
}

h3 {
    font-size: 20px;
}
"""

with open("source/_static/custom.css", "w") as f:
    f.write(css_content)

# Create custom template to skip unwanted members
layout_content = """{% extends "!layout.html" %}

{% block extrahead %}
{{ super() }}
<style>
    body, .body {
        font-size: 16px !important;
    }
    pre, code {
        font-size: 14px !important;
    }
    h1 { font-size: 32px !important; }
    h2 { font-size: 24px !important; }
    h3 { font-size: 20px !important; }
</style>
{% endblock %}
"""

with open("source/_templates/layout.html", "w") as f:
    f.write(layout_content)

# Create fake module stubs for imports
os.makedirs(os.path.join("source", "_mock"), exist_ok=True)

# Create __init__.py to make it a package
with open(os.path.join("source", "_mock", "__init__.py"), "w") as f:
    f.write("# Mock package\n")

# Create chatbot module stub
os.makedirs(os.path.join("source", "_mock", "chatbot"), exist_ok=True)
with open(os.path.join("source", "_mock", "chatbot", "__init__.py"), "w") as f:
    f.write("# Mock chatbot package\n")

# Create config stub
with open(os.path.join("source", "_mock", "chatbot", "config.py"), "w") as f:
    f.write("""
# Mock config module for documentation
class OpenAIConfig:
    pass

class ClaudeConfig:
    pass

class GeminiConfig:
    pass

class MistralConfig:
    pass

class GrokConfig:
    pass

def get_api_key():
    pass
""")

# Create utils and paths stubs
os.makedirs(os.path.join("source", "_mock", "chatbot", "utils"), exist_ok=True)
with open(os.path.join("source", "_mock", "chatbot", "utils", "__init__.py"), "w") as f:
    f.write("# Mock utils package\n")

with open(os.path.join("source", "_mock", "chatbot", "utils", "utils.py"), "w") as f:
    f.write("""
# Mock utils module
class DatabaseOps:
    pass

def create_folder():
    pass
""")

with open(os.path.join("source", "_mock", "paths.py"), "w") as f:
    f.write("""
# Mock paths module
DATASETS_DIR = "mock/path/to/datasets"
""")

# Also handle crawler for main.py
with open(os.path.join("source", "_mock", "chatbot", "crawler.py"), "w") as f:
    f.write("""
# Mock crawler module
class Crawler:
    pass
""")

# Create rag module stubs
os.makedirs(os.path.join("source", "_mock", "chatbot", "rag"), exist_ok=True)
with open(os.path.join("source", "_mock", "chatbot", "rag", "__init__.py"), "w") as f:
    f.write("# Mock rag package\n")

with open(os.path.join("source", "_mock", "chatbot", "rag", "openai_rag.py"), "w") as f:
    f.write("""
# Mock OpenAI RAG module
class OpenAIRAG:
    pass
""")

# Add mock directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "source", "_mock"))

# Update conf.py to include custom settings
conf_additional = """
# Add custom static path and CSS
html_static_path = ['_static']
html_css_files = ['custom.css']
templates_path = ['_templates']

# Add the project root to the path for imports
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../_mock'))

# Mock imports for modules that can't be imported
autodoc_mock_imports = [
    'chatbot', 
    'paths', 
    'utils',
    'openai',
    'anthropic',
    'google',
    'mistral',
    'cohere',
    'deepseek',
    'grok',
    'requests',
    'numpy',
    'pandas',
    'sklearn',
    'torch',
    'transformers'
]

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': True,
    'show-inheritance': True,
    'exclude-members': '__dict__,__weakref__,__module__,__annotations__'
}

def skip_unwanted(app, what, name, obj, skip, options):
    \"\"\"Skip unwanted members in documentation.\"\"\"
    # List of names to always skip
    skip_names = [
        '__dict__', 
        '__weakref__', 
        '__module__', 
        '__abstractmethods__',
        '__annotations__',
        '__dataclass_fields__',
        '__dataclass_params__',
        '__doc__',
        '__hash__',
        '__slots__',
        '_abc_impl',
        '__pycache__'
    ]
    
    # Skip all double-underscore methods except a select few
    if name.startswith('__') and name.endswith('__'):
        # Keep only specific special methods if needed
        keep_special = ['__init__', '__call__', '__str__', '__repr__', '__len__']
        return name not in keep_special
    
    # Skip __pycache__ related modules
    if '__pycache__' in name:
        return True
    
    # Skip specific named members
    if name in skip_names:
        return True
    
    # Honor the default skip for other members
    return skip

def setup(app):
    # Connect the skip function to the autodoc-skip-member event
    app.connect('autodoc-skip-member', skip_unwanted)
"""

# Check if conf.py exists and add our custom settings
conf_path = "source/conf.py"
if os.path.exists(conf_path):
    with open(conf_path, "r") as f:
        content = f.read()
    
    # Replace any existing mock_imports
    if "autodoc_mock_imports" in content:
        import re
        content = re.sub(r'autodoc_mock_imports\s*=\s*\[.*?\]', 'autodoc_mock_imports = []', content, flags=re.DOTALL)
    
    # Append our additional settings if they're not already there
    if "skip_unwanted" not in content:
        with open(conf_path, "w") as f:
            f.write(content)
            f.write(conf_additional)
    else:
        print("skip_unwanted already in conf.py, not modifying it")
else:
    print("Warning: conf.py not found. Creating a basic one.")
    # Create a minimal conf.py if it doesn't exist
    with open(conf_path, "w") as f:
        f.write("""
# Configuration file for the Sphinx documentation builder.
import os
import sys

project = 'DataVerse Chatbot'
copyright = '2025, Ali Elneklawy'
author = 'Ali Elneklawy'
release = '2.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/__pycache__']

html_theme = 'sphinx_rtd_theme'
""")
        f.write(conf_additional)

# Create an __init__.py file in the source directory to make it a package
with open("source/__init__.py", "w") as f:
    f.write("# Documentation package")

# Create temporary __init__.py files if they don't exist
# to make sure all directories are treated as packages
for root, dirs, files in os.walk("../src"):
    for dir_name in dirs:
        init_file = os.path.join(root, dir_name, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Auto-generated package file for documentation\n")
            print(f"Created missing __init__.py at {init_file}")

# Run sphinx-apidoc to generate module documentation
subprocess.run([
    "sphinx-apidoc", 
    "-o", "source",          # Output to source directory
    "-f",                    # Force overwrite existing files
    "--separate",            # Generate separate file for each module
    "--private",             # Include private modules (_*.py)
    "--module-first",        # Put module before submodule documentation
    "-d", "__pycache__",     # Exclude __pycache__ directories
    "../src"                 # Path to your module
])

# Use sphinx-build directly instead of make
sphinx_env = os.environ.copy()
sphinx_env["PYTHONPATH"] = os.pathsep.join([
    project_root,
    os.path.join(project_root, 'src'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "source", "_mock"),
    sphinx_env.get("PYTHONPATH", "")
])

print(f"Using PYTHONPATH: {sphinx_env['PYTHONPATH']}")

subprocess.run([
    "sphinx-build",
    "-b", "html",            # Build HTML
    "-a",                    # Write all files
    "-E",                    # Don't use a saved environment, always read all files
    "source",                # Source directory
    "build/html"             # Output directory
], env=sphinx_env)

print("Documentation generation complete. Check the docs/build/html directory.")