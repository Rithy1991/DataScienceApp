import os
import re

def update_nav():
    # Update pages
    for f in os.listdir("pages"):
        if not f.endswith(".py"): continue
        
        # specific handling based on current filename
        # Pattern: N_Name.py
        match = re.match(r"(\d+)_", f)
        if match:
            num = match.group(1)
            path = os.path.join("pages", f)
            with open(path, "r") as file:
                content = file.read()
            
            # Regex to find page_navigation call
            # It might look like page_navigation("3") or page_navigation('3')
            new_content = re.sub(r'page_navigation\s*\(\s*["\']\d+["\']\s*\)', f'page_navigation("{num}")', content)
            
            if content != new_content:
                with open(path, "w") as file:
                    file.write(new_content)
                print(f"Updated {f} to ID {num}")
            else:
                # If not found, append it (safety)
                if "page_navigation" not in content:
                   # Don't blind append, might be missing imports.
                   print(f"Skipping {f} - no nav found")
                else:
                   print(f"No change needed for {f} (or regex mismatch)")

    # Update app.py
    if os.path.exists("app.py"):
        with open("app.py", "r") as file:
            content = file.read()
        
        # app.py should be "0"
        new_content = re.sub(r'page_navigation\s*\(\s*["\']\d+["\']\s*\)', 'page_navigation("0")', content)
        if content != new_content:
            with open("app.py", "w") as file:
                file.write(new_content)
            print("Updated app.py to ID 0")

if __name__ == "__main__":
    update_nav()
