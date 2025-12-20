#!/usr/bin/env python3
"""
Scrape Korean medical law content from various sources
"""

import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import time
import re

# Korean medical law sources
SOURCES = {
    "law_go_kr": {
        "name": "국가법령정보센터",
        "base_url": "https://www.law.go.kr",
        "laws": [
            {"name": "의료법", "id": "000003"},
            {"name": "약사법", "id": "000007"},
            {"name": "감염병의 예방 및 관리에 관한 법률", "id": "011113"},
            {"name": "의료기기법", "id": "006799"},
            {"name": "생명윤리 및 안전에 관한 법률", "id": "009100"},
            {"name": "응급의료에 관한 법률", "id": "004260"},
            {"name": "정신건강증진 및 정신질환자 복지서비스 지원에 관한 법률", "id": "014224"},
            {"name": "혈액관리법", "id": "006010"},
            {"name": "장기등 이식에 관한 법률", "id": "005859"},
            {"name": "국민건강보험법", "id": "005854"},
        ]
    }
}

def fetch_law_content(law_id, law_name):
    """Fetch law content from 국가법령정보센터"""
    # API endpoint for law content
    url = f"https://www.law.go.kr/DRF/lawService.do?OC=ormastes&target=law&type=JSON&ID={law_id}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data
    except Exception as e:
        print(f"Error fetching {law_name}: {e}")

    return None

def parse_law_articles(law_data):
    """Parse law articles from API response"""
    articles = []

    if not law_data:
        return articles

    # Extract articles from the law data
    try:
        law_info = law_data.get('법령', {})
        article_list = law_info.get('조문', [])

        for article in article_list:
            article_info = {
                'article_no': article.get('조문번호', ''),
                'article_title': article.get('조문제목', ''),
                'article_content': article.get('조문내용', ''),
            }
            articles.append(article_info)
    except Exception as e:
        print(f"Error parsing articles: {e}")

    return articles

def generate_qa_from_article(law_name, article):
    """Generate Q&A pairs from a law article (template-based)"""
    qa_pairs = []

    article_no = article.get('article_no', '')
    article_title = article.get('article_title', '')
    article_content = article.get('article_content', '')

    if not article_content:
        return qa_pairs

    # Template 1: What does Article X say?
    qa_pairs.append({
        'question': f"{law_name} 제{article_no}조({article_title})의 내용은 무엇입니까?",
        'answer': article_content,
        'law_name': law_name,
        'article_no': article_no,
        'article_title': article_title,
        'type': 'article_content'
    })

    # Template 2: According to law, what is...
    if article_title:
        qa_pairs.append({
            'question': f"{law_name}에 따르면, '{article_title}'에 관한 규정은 무엇입니까?",
            'answer': article_content,
            'law_name': law_name,
            'article_no': article_no,
            'article_title': article_title,
            'type': 'topic_based'
        })

    return qa_pairs

def scrape_all_laws():
    """Scrape all configured laws"""
    all_qa_pairs = []

    for source_id, source_info in SOURCES.items():
        print(f"\n=== {source_info['name']} ===")

        for law_info in source_info['laws']:
            law_name = law_info['name']
            law_id = law_info['id']

            print(f"Fetching: {law_name}...")

            law_data = fetch_law_content(law_id, law_name)

            if law_data:
                articles = parse_law_articles(law_data)
                print(f"  Found {len(articles)} articles")

                for article in articles:
                    qa_pairs = generate_qa_from_article(law_name, article)
                    all_qa_pairs.extend(qa_pairs)

            time.sleep(1)  # Rate limiting

    return all_qa_pairs

def main():
    output_dir = Path("/home/ormastes/dev/pub/medgemma_korean/data/raw/korean_datasets/korean_medical_law_scraped")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scraping Korean medical laws...")
    qa_pairs = scrape_all_laws()

    print(f"\nTotal Q&A pairs generated: {len(qa_pairs)}")

    # Save to JSONL
    output_file = output_dir / "korean_medical_law_qa.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    print(f"Saved to: {output_file}")

    # Print sample
    if qa_pairs:
        print("\n=== Sample Q&A ===")
        sample = qa_pairs[0]
        print(f"Q: {sample['question'][:100]}...")
        print(f"A: {sample['answer'][:200]}...")

if __name__ == "__main__":
    main()
