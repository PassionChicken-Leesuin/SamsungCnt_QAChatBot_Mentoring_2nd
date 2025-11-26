import requests
import xml.etree.ElementTree as ET
import pandas as pd

def fetch_google_news_rss(query="산업 안전 사고", max_items=20):
    base_url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = []

    for item in root.findall(".//item")[:max_items]:
        items.append({
            "title": item.findtext("title"),
            "link": item.findtext("link"),
            "pubDate": item.findtext("pubDate"),
            "description": item.findtext("description")  # ← 요약/스니펫 추가
        })

    return pd.DataFrame(items)
