import os

for filename in os.listdir("data"):
    if filename.endswith(".md"):
        if filename in ['REST APIs.md','Web Framework: FastAPI.md','Hybrid RAG with TypeSense.md']:
            continue
        path = os.path.join("data", filename)
        with open(path, "r", encoding="utf-8") as f:
            if "404 - Not found" in f.read():
                os.remove(path)
                print(f"Deleted: {filename}")