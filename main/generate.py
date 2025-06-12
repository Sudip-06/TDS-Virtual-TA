import json
import os
from typing import List, Dict, Any
import requests
from pinecone import Pinecone, ServerlessSpec
import base64
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


def encode_image_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = ai_pipe_embedding(query)
    search_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    results = []
    for match in search_response.matches:
        try:
            results.append({
                "combined_text": match.metadata["combined_text"],
                "link":f"https://discourse.onlinedegree.iitm.ac.in/t/{'-'.join(match.metadata["topic_title"].split(' '))}/{int(match.metadata["topic_id"])}",
                "score":match.score
            })
        except:
            try:
                results.append({
                    "link": match.metadata["link"],
                    "combined_text": match.metadata["combined_text"],
                    "score":match.score
                })
            except:
                continue
    return results

def ai_pipe_chat_completion(messages: List[Dict[str, Any]]) -> str:
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def generate_answer(query: str, context_texts: List[str], image_url: str = None, image_base64: str = None) -> str:
    context = "\n\n---\n\n".join(context_texts)
    messages = [
        {"role": "system", 
         "content": "You are a helpful assistant that answers questions based on forum discussions."},
    ]
    if image_base64 or image_url:
        print(image_url)
        messages.append({
            "role": "user",
            "content": f"Based on these forum excerpts:\n\n{context}\n\nImage: {image_url}\n\nQuestion: {query}"
        })
    else:
        messages.append({
            "role": "user",
            "content": f"Based on these forum excerpts:\n\n{context}\n\nQuestion: {query}"
        })
    return ai_pipe_chat_completion(messages)


def output(query,image_url=None,k=3):
    results = semantic_search(query, top_k=k)
    # if not results or (results[0].get("score", 0) < 0.4):
    #     return {"answer": "Sorry, the answer is not available in the discourse.", "links": []}
    # print(results[0]["score"])
    # print("\nTop search results:")
    # for i, res in enumerate(results, 1):
    #     print(f"Link: {res['link']}")
    #     print(f"Content snippet: {res['combined_text'][:500]}...\n")
    context_texts = [res["combined_text"] for res in results]
    answer = generate_answer(query, context_texts,image_url=image_url)
    return {'answer':answer,'links':[{"url":res['link'],'text':res['combined_text'][:100]} for res in results]}

if __name__=='__main__':
    query = "I know Docker but have not used Podman. Should I use Docker for this not available in the course? "
    # results = semantic_search(query, top_k=3)

    # print("\nTop search results:")
    # for i, res in enumerate(results, 1):
    #     print(f"Link: {res['link']} {res['score']}")
    #     print(f"Content snippet: {res['combined_text'][:500]}...\n")

    # context_texts = [res["combined_text"] for res in results]
    # answer = generate_answer(query, context_texts)
    # print("\nGenerated Answer:\n", answer)
    print(output(query))