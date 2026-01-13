#!/usr/bin/env python3
"""Replace Streamlit's deprecated `use_container_width` with the new `width`.

Mapping:
- use_container_width=True  -> width="stretch"
- use_container_width=False -> width="content"

This script is intentionally conservative and only replaces explicit True/False.
"""

import re
from pathlib import Path


workspace_root = Path(__file__).parent


_RE_TRUE = re.compile(r"\buse_container_width\s*=\s*True\b")
_RE_FALSE = re.compile(r"\buse_container_width\s*=\s*False\b")


def fix_file(filepath: Path) -> bool:
    """Replace `use_container_width` occurrences in a single file."""
    try:
        content = filepath.read_text(encoding="utf-8")
        original_content = content

        content = _RE_TRUE.sub('width="stretch"', content)
        content = _RE_FALSE.sub('width="content"', content)

        if content != original_content:
            filepath.write_text(content, encoding="utf-8")
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
        # Skip this script itself and any environments/caches.
        py_file_str = str(py_file)
        if (
            py_file.name == "fix_deprecation.py"
            or "__pycache__" in py_file_str
            or "/.venv/" in py_file_str
            or "/venv/" in py_file_str
        ):
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
