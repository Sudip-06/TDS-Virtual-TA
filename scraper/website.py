import os
import sys
import re
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from time import sleep
import shutil
import html2text

def clean_text(raw_text: str) -> str:
    cleaned = re.sub(r'window\.\$docsify\s*=\s*\{.*?\};', '', raw_text, flags=re.DOTALL)
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    while lines and lines[0] == '':
        lines.pop(0)
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)

def clean_text_with_links(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a['href'].strip()
        link_text = a.get_text(strip=True)
        if href and link_text:
            a.string = f"{link_text} [{href}]"
        elif href:
            a.string = f"[{href}]"
    for img in soup.find_all("img", src=True):
        alt = img.get("alt", "").strip()
        src = img["src"].strip()
        img.replace_with(f"![{alt}]({src})")
    text = soup.get_text(separator=" ")
    return clean_text(text)

if '-del' in sys.argv:
    print("It will delete the file data/data.md")
    assert(False)
    try:
        shutil.rmtree("data")
        print("File Deleted")
    except FileNotFoundError:
        print("File not found")
    print()

if not os.path.exists("data"):
    p = sync_playwright().start()

    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://tds.s-anand.net/#/2025-01/", wait_until="networkidle")

    os.makedirs("data", exist_ok=True)
    while True:
        html_body = page.eval_on_selector("body", "el => { const clone = el.cloneNode(true); clone.querySelector('.sidebar')?.remove(); return clone.innerHTML; }")

        # formatted_text = clean_text_with_links(html_body)

        markdown = html2text.html2text(html_body)

        title = page.title().replace("/", "_").replace("\\", "_") 
        if os.path.exists(f'data/{title}'):
            break
        file = open(f"data/{title}.md", "w", encoding="utf-8")
        url = page.url
        file.write(f"Current Page URL: {url}\n\n{markdown}")
        file.close()
        print(f"{title} Written")

        page.locator(".pagination-item-label").last.click()
        page.wait_for_load_state("networkidle")

        sleep(1)

    browser.close()
    p.stop()
else:
    print("File already exists")