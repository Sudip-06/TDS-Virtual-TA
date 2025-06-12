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

def process_posts(filename: str) -> Dict[int, Dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as f:
        posts_data = json.load(f)
    topics = {}
    for post in posts_data:
        topic_id = post["topic_id"]
        if topic_id not in topics:
            topics[topic_id] = {
                "topic_title": ' '.join((post.get("topic_slug", "")).split('-')),
                "posts": []
            }
        topics[topic_id]["posts"].append(post)
    for topic in topics.values():
        topic["posts"].sort(key=lambda p: p["post_number"])
    return topics

def build_thread_map(posts: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    thread_map = {}
    for post in posts:
        parent = post.get("reply_to_post_number")
        if parent not in thread_map:
            thread_map[parent] = []
        thread_map[parent].append(post)
    return thread_map

def extract_thread(root_num: int, posts: List[Dict[str, Any]], thread_map: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    thread = []
    def collect_replies(post_num):
        post = next(p for p in posts if p["post_number"] == post_num)
        thread.append(post)
        for reply in thread_map.get(post_num, []):
            collect_replies(reply["post_number"])
    collect_replies(root_num)
    return thread

def embed_and_index_threads(topics: Dict[int, Dict[str, Any]], batch_size: int = 100):
    vectors = []
    for topic_id, topic_data in tqdm(topics.items()):
        posts = topic_data["posts"]
        topic_title = topic_data["topic_title"]
        thread_map = build_thread_map(posts)
        root_posts = thread_map.get(None, [])
        for root_post in root_posts:
            thread = extract_thread(root_post["post_number"], posts, thread_map)
            combined_text = f"Topic: {topic_title}\n\n"
            combined_text += "\n\n---\n\n".join(
                post["content"].strip() for post in thread
            )
            embedding = ai_pipe_embedding(combined_text)
            vector = {
                "id": f"{topic_id}_{root_post['post_number']}",
                "values": embedding,
                "metadata": {
                    "topic_id": topic_id,
                    "topic_title": topic_title,
                    "root_post_number": root_post["post_number"],
                    "post_numbers": [str(p["post_number"]) for p in thread],
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

if False:
    topics = process_posts("posts.json")
    embed_and_index_threads(topics)
    print("Indexing complete")
    exit()
