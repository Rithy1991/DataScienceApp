#!/usr/bin/env python3
"""
Script to replace deprecated use_container_width parameter with new width parameter
"""
import os
import re
from pathlib import Path

# Get the workspace root directory
workspace_root = Path(__file__).parent

def fix_file(filepath):
    """Replace use_container_width in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace use_container_width=True with use_container_width=True
        content = content.replace("use_container_width=True", "use_container_width=True")
        
        # Replace use_container_width=False with width='content'
        content = content.replace("use_container_width=False", "width='content'")
        
        # Only write if there were changes
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Find and fix all Python files."""
    files_modified = []
    
    # Find all Python files
    for py_file in workspace_root.rglob("*.py"):
        # Skip this script itself and any __pycache__ or .venv directories
        if (py_file.name == "fix_deprecation.py" or 
            "__pycache__" in str(py_file) or
            ".venv" in str(py_file) or
            "venv" in str(py_file)):
            continue
        
        if fix_file(py_file):
            files_modified.append(py_file)
            print(f"✓ Modified: {py_file.relative_to(workspace_root)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Modified {len(files_modified)} files")
    print(f"{'='*60}")
    
    if files_modified:
        print("\nModified files:")
        for f in sorted(files_modified):
            print(f"  - {f.relative_to(workspace_root)}")
    else:
        print("\nNo files needed modification.")

if __name__ == "__main__":
    main()
