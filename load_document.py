
# load_document.py
from pathlib import Path

file_path = Path("documents/school_handbook.txt")

with open(file_path, "r", encoding="utf-8") as file:
    document = file.read()

print(document)