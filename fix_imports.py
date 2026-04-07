import os
import re

root = 'src'
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.endswith('.py'):
            filepath = os.path.join(dirpath, f)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            # Replace `from core...` with `from src.core...` (same for experience, genome, training, tools, cli)
            new_content = re.sub(r'(?<!\w)(from|import)\s+(core|experience|genome|training|tools|cli)(?=\s|\.)', r'\1 src.\2', content)
            # Also handle `import core` without `from`? Already covered.
            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"Updated: {filepath}")