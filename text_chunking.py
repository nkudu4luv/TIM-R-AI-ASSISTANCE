from pathlib import Path

file_path = Path("documents/school_handbook.txt")
document = file_path.read_text(encoding="utf-8")

words = document.split()
chunk_size = 50
chunks = []

for start in range(0, len(words), chunk_size):
    chunk = words[start:start + chunk_size]
    chunks.append(" ".join(chunk))

print("Total chunks:", len(chunks))

for number, chunk in enumerate(chunks, start=1):
    print(f"\nCHUNK {number}")
    print(chunk)