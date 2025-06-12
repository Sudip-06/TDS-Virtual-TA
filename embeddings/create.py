import os
import json
from bs4 import BeautifulSoup

def load_all_json(folder_path):
    all_posts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)['post_stream']['posts']
                for j in data:
                    j['content'] = BeautifulSoup(j['cooked'],"html.parser").get_text().strip()
                all_posts.extend(data)
    return all_posts

all_posts = load_all_json("data_discourse")

with open("posts.json", "w", encoding="utf-8") as f:
    json.dump(all_posts, f, ensure_ascii=False, indent=2)