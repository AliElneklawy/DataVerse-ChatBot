# create_manual_docs.py
import os
import sys
import glob

def generate_rst_for_file(filepath, output_dir):
    """Generate an RST file for a Python module."""
    module_name = os.path.basename(filepath).replace('.py', '')
    if module_name == '__init__':
        return None  # Skip __init__.py files
        
    package_path = os.path.dirname(filepath)
    package_name = os.path.basename(package_path)
    
    # Create a suitable filename for the rst file
    if package_name == 'src':
        rst_filename = f"{module_name}.rst"
    else:
        rst_filename = f"{package_name}.{module_name}.rst"
    
    rst_path = os.path.join(output_dir, rst_filename)
    
    # Read the Python file to extract docstrings and function definitions
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
        except UnicodeDecodeError:
            content = "Unable to read file due to encoding issues."
    
    # Extract class and function definitions using simple parsing
    classes = []
    functions = []
    current_indentation = 0
    current_definition = ""
    in_definition = False

    for line in content.split('\n'):
        stripped = line.strip()
        
        # Very basic detection of class and function definitions
        if stripped.startswith('class ') and stripped.endswith(':'):
            classes.append(stripped[6:-1].split('(')[0].strip())
        elif stripped.startswith('def ') and stripped.endswith(':'):
            functions.append(stripped[4:-1].split('(')[0].strip())
    
    # Create RST content
    title = f"{module_name} module"
    rst_content = f"""
{title}
{'=' * len(title)}

.. py:module:: {package_name}.{module_name}

Module Description
-----------------

*Documentation for this module was auto-generated.*

"""

    if classes:
        rst_content += "\nClasses\n-------\n\n"
        for cls in classes:
            rst_content += f"* ``{cls}``\n"
    
    if functions:
        rst_content += "\nFunctions\n---------\n\n"
        for func in functions:
            rst_content += f"* ``{func}``\n"
    
    # Write the RST file
    with open(rst_path, 'w', encoding='utf-8') as f:
        f.write(rst_content)
    
    return rst_filename

def generate_package_rst(package_dir, output_dir, package_name=None):
    """Generate RST for a package directory."""
    if package_name is None:
        package_name = os.path.basename(package_dir)
    
    # Skip __pycache__ and other non-Python dirs
    if package_name.startswith('__') or package_name.startswith('.'):
        return None
    
    # Create a suitable filename for the package rst
    rst_filename = f"{package_name}.rst"
    rst_path = os.path.join(output_dir, rst_filename)
    
    # Get all Python files in this package
    py_files = glob.glob(os.path.join(package_dir, '*.py'))
    subpackages = []
    
    # Find subpackages (directories with __init__.py)
    for item in os.listdir(package_dir):
        subpackage_dir = os.path.join(package_dir, item)
        if os.path.isdir(subpackage_dir) and not item.startswith('.') and not item.startswith('__'):
            if os.path.exists(os.path.join(subpackage_dir, '__init__.py')):
                subpackages.append(item)
    
    # Generate RST for all Python files in this package
    module_rst_files = []
    for py_file in py_files:
        if os.path.basename(py_file) != '__init__.py':
            rst_file = generate_rst_for_file(py_file, output_dir)
            if rst_file:
                module_rst_files.append(rst_file.replace('.rst', ''))
    
    # Generate RST for all subpackages
    subpackage_rst_files = []
    for subpackage in subpackages:
        subpackage_dir = os.path.join(package_dir, subpackage)
        rst_file = generate_package_rst(subpackage_dir, output_dir, f"{package_name}.{subpackage}")
        if rst_file:
            subpackage_rst_files.append(rst_file.replace('.rst', ''))
    
    # Create RST content for the package
    title = f"{package_name} package"
    rst_content = f"""
{title}
{'=' * len(title)}

.. py:module:: {package_name}

Package Description
------------------

*Documentation for this package was auto-generated.*

"""
    
    if subpackage_rst_files:
        rst_content += "\nSubpackages\n-----------\n\n"
        rst_content += ".. toctree::\n    :maxdepth: 1\n\n"
        for subpackage in subpackage_rst_files:
            rst_content += f"    {subpackage}\n"
    
    if module_rst_files:
        rst_content += "\nModules\n-------\n\n"
        rst_content += ".. toctree::\n    :maxdepth: 1\n\n"
        for module in module_rst_files:
            rst_content += f"    {module}\n"
    
    # Write the RST file
    with open(rst_path, 'w', encoding='utf-8') as f:
        f.write(rst_content)
    
    return rst_filename

def update_index_rst(output_dir, top_level_packages):
    """Update the index.rst file to include top-level packages."""
    index_path = os.path.join(output_dir, 'index.rst')
    
    # Read existing content
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = """
Welcome to DataVerse ChatBot documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
    
    # Find the toctree and add packages
    lines = content.split('\n')
    toctree_found = False
    for i, line in enumerate(lines):
        if ':caption: Contents:' in line:
            toctree_found = True
            # Add an empty line after the caption if needed
            if i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith('   '):
                lines.insert(i+1, '')
            
            # Add packages after the empty line
            for j, package in enumerate(top_level_packages):
                # Check if package is already in toctree
                if any(f"   {package}" in l for l in lines):
                    continue
                lines.insert(i+j+2, f"   {package}")
            break
    
    # If no toctree with caption found, add one before the indices
    if not toctree_found:
        indices_index = -1
        for i, line in enumerate(lines):
            if line.strip() == "Indices and tables":
                indices_index = i
                break
        
        if indices_index > 0:
            # Add the package section before indices
            packages_section = [
                "",
                "Packages",
                "========",
                "",
                ".. toctree::",
                "   :maxdepth: 2",
                ""
            ]
            for package in top_level_packages:
                packages_section.append(f"   {package}")
            packages_section.append("")
            
            lines = lines[:indices_index] + packages_section + lines[indices_index:]
    
    # Write the updated index.rst
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def main():
    # Get source directory (assumes this script is in docs directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
    output_dir = os.path.join(script_dir, 'source')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate RST for all packages in src
    top_level_packages = []
    
    # First, handle the src directory itself for top-level modules
    py_files = glob.glob(os.path.join(src_dir, '*.py'))
    for py_file in py_files:
        if os.path.basename(py_file) != '__init__.py':
            rst_file = generate_rst_for_file(py_file, output_dir)
            if rst_file:
                top_level_packages.append(rst_file.replace('.rst', ''))
    
    # Then handle packages inside src
    for item in os.listdir(src_dir):
        package_dir = os.path.join(src_dir, item)
        if os.path.isdir(package_dir) and not item.startswith('.') and not item.startswith('__'):
            if os.path.exists(os.path.join(package_dir, '__init__.py')):
                rst_file = generate_package_rst(package_dir, output_dir, item)
                if rst_file:
                    top_level_packages.append(rst_file.replace('.rst', ''))
    
    # Update index.rst to include top-level packages
    update_index_rst(output_dir, top_level_packages)
    
    print(f"Generated documentation for {len(top_level_packages)} top-level packages/modules.")

if __name__ == "__main__":
    main()
