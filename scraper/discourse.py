import os
import sys
import re
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from time import sleep
import shutil
import html2text
import requests
from datetime import datetime
import random
import json

if '-del' in sys.argv:
    print("It will delete the file")
    assert(False)
    try:
        shutil.rmtree("data_discourse")
        print("File Deleted")
    except FileNotFoundError:
        print("File not found")
    print()

def fetch(url,session):
    res = session.get(url)
    if res.status_code==200:
        return res.json()
    print(f"Failed {url}: {res.status_code}")
    return None

if not os.path.exists("data_discourse"):
    session = requests.session()
    session.cookies.update({
        '_forum_session':os.environ.get('COOKIE'),
        '_fbp':os.environ.get('_fbp'),
        '_ga_5HTJMW67XK':os.environ.get('_ga_5HTJMW67XK'),
        '_ga_08NPRH5L4M':os.environ.get('_ga_08NPRH5L4M'),
        '_ga':os.environ.get('_ga'),
        '_gcl_au':os.environ.get('_gcl_au'),
        '_t':os.environ.get('_t')
    })

    start = datetime(2025, 1, 1)
    end = datetime(2025, 4, 14)
    os.makedirs("data_discourse", exist_ok=True)
    def write(id):
        file = open(f'data_discourse/{id}.json',"w",encoding='utf-8')
        data = fetch(f'https://discourse.onlinedegree.iitm.ac.in/t/{id}.json',session)
        json.dump(data,file,ensure_ascii=False,indent=2)
        file.close()
        return

    page = 0
    while True:
        data = fetch(f'https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json?page={page}',session)
        
        try:
            for i in data['topic_list']['topics']:
                if start<=datetime.strptime(i['created_at'],"%Y-%m-%dT%H:%M:%S.%fZ")<=end:
                    write(i['id'])
                    sleep(random.randint(2,3))
            page += 1
            sleep(random.randint(2,3))
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

else:
    print("File already exists")