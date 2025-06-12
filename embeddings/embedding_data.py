import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
import requests
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

AI_PIPE_API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY = os.getenv("PINE_KEY")

AI_PIPE_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

pinecone = Pinecone(api_key=PINECONE_API_KEY)

index_name = "discourse-embeddings"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(index_name)

HEADERS = {
    "Authorization": f"{AI_PIPE_API_KEY}",
    "Content-Type": "application/json",
}

def ai_pipe_embedding(text: str) -> List[float]:
    url = f"{AI_PIPE_BASE_URL}/embeddings"
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def embed_and_index_pages(pages: Dict[int, Dict[str, str]], batch_size: int = 100):
    vectors = []
    for page_id, page_data in tqdm(pages.items()):
        link = page_data["link"]
        topic_title = page_data["topic_title"]
        content = page_data["content"]

        combined_text = f"Link: {link}\nTitle: {topic_title}\n\n{content.strip()}"
        embedding = ai_pipe_embedding(combined_text)
        vector = {
            "id": f"page_{page_id}",
            "values": embedding,
            "metadata": {
                "link": link,
                "topic_title": topic_title,
                "combined_text": combined_text
            }
        }
        vectors.append(vector)
        if len(vectors) >= batch_size:
            try:
                index.upsert(vectors=vectors)
            except Exception as e:
                import random
                k = random.randint(1,100000)
                with open(f'{k}.txt', 'w') as f:
                    f.write(str(e))
            vectors = []
    if vectors:
        index.upsert(vectors=vectors)

def load_pages_from_folder(folder_path: str) -> dict:
    pages = {}
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                continue
            pages[i] = {
                "link": lines[0].strip()[18:],
                "topic_title": os.path.splitext(filename)[0],
                "content": ''.join(lines[1:]).strip()
            }
    return pages

if False:
    topics = load_pages_from_folder("data")
    embed_and_index_pages(topics)
    print("Indexing complete")
    exit()
