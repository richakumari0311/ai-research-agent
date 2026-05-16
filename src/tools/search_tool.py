import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def search_web(query: str) -> str:
    if not SERPER_API_KEY:
        return "SERPER_API_KEY not set in .env file."

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": 5
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Search failed with status {response.status_code}"

    data = response.json()
    results = data.get("organic", [])

    if not results:
        return "No search results found."

    output = []
    for i, result in enumerate(results):
        title = result.get("title", "No title")
        snippet = result.get("snippet", "No snippet")
        link = result.get("link", "")
        output.append(f"{i+1}. {title}\n   {snippet}\n   URL: {link}")

    return "\n\n".join(output)