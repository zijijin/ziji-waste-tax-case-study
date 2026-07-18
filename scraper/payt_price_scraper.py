"""
PAYT Bag-Price Harvester — run locally (VS Code / Claude Code), NOT in a sandbox.

For each Massachusetts PAYT town, searches the web for the town's official
trash/PAYT page, fetches candidate municipal pages, and extracts dollar amounts
that appear near bag-related keywords. Output is a CANDIDATES file with the
surrounding text snippet and source URL for every hit, so each price can be
human-verified before analysis. This is a human-in-the-loop harvester, not a
blind scraper — verification is part of the method.

Setup:
    pip install requests beautifulsoup4 ddgs
Run:
    python payt_price_scraper.py --towns payt_towns.csv --out payt_price_candidates.csv
    (optionally --limit 30 to do the largest 30 towns first)

Then upload payt_price_candidates.csv back to the Claude conversation.
"""
import argparse
import csv
import os
import re
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS          # package name changed from duckduckgo_search
except ImportError:
    from duckduckgo_search import DDGS

HEADERS = {"User-Agent": "Mozilla/5.0 (research; PAYT fee-schedule study; contact via claude.ai project)"}
BAG_WORDS = r"(?:bag|bags|sticker|stickers|tag|tags|overflow|payt|pay.as.you.throw)"
SIZE_WORDS = r"(?:\d{1,2}\s*[- ]?gal(?:lon)?s?|small|large|kitchen|barrel)"
PRICE = r"\$ ?\d{1,2}(?:\.\d{2})?"

# window of characters around a price to search for bag context
CTX = 160

def official_candidates(town: str, max_results: int = 6):
    """Search for the town's trash/PAYT page; prefer municipal domains."""
    queries = [
        f"{town} MA pay as you throw trash bag price",
        f"{town} Massachusetts trash bag fee DPW",
    ]
    urls = []
    with DDGS() as ddgs:
        for q in queries:
            try:
                for r in ddgs.text(q, max_results=max_results):
                    u = r.get("href") or r.get("url") or ""
                    if not u:
                        continue
                    urls.append(u)
            except Exception as e:
                print(f"  search error for {town}: {e}")
            time.sleep(1.0)
    # rank: municipal-looking domains first (townofX.org, Xma.gov, X-ma.gov, .ma.us)
    tkey = town.lower().replace(" ", "").replace("-", "")
    def score(u):
        ul = u.lower()
        s = 0
        if tkey[:8] in ul: s += 2
        if any(p in ul for p in (".gov", ".ma.us", "townof", "cityof")): s += 2
        if any(p in ul for p in ("trash", "solid-waste", "solidwaste", "recycl", "payt", "transfer")): s += 1
        if any(p in ul for p in ("patch.com", "wiki", "facebook", "yelp")): s -= 2
        return -s
    seen, ranked = set(), []
    for u in sorted(urls, key=score):
        if u not in seen:
            seen.add(u)
            ranked.append(u)
    return ranked[:4]

def extract_prices(text: str):
    """Yield (price_str, context) where a $ amount sits near bag words."""
    for m in re.finditer(PRICE, text):
        lo, hi = max(0, m.start() - CTX), min(len(text), m.end() + CTX)
        window = text[lo:hi]
        if re.search(BAG_WORDS, window, re.I):
            ctx = re.sub(r"\s+", " ", window).strip()
            yield m.group(0), ctx

def fetch_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer"]):
        t.decompose()
    return soup.get_text(" ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--towns", default=os.path.join(SCRIPT_DIR, "payt_towns.csv"))
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "payt_price_candidates.csv"))
    ap.add_argument("--limit", type=int, default=None, help="only first N towns (file is sorted largest-first)")
    args = ap.parse_args()

    with open(args.towns, newline="", encoding="utf-8-sig") as f:
        towns = [row["municipality"] for row in csv.DictReader(f)]
    if args.limit:
        towns = towns[: args.limit]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["municipality", "price_found", "context_snippet", "source_url"])
        for i, town in enumerate(towns, 1):
            print(f"[{i}/{len(towns)}] {town}")
            try:
                for url in official_candidates(town):
                    try:
                        text = fetch_text(url)
                    except Exception as e:
                        print(f"    fetch fail {url[:60]}: {e}")
                        continue
                    hits = list(extract_prices(text))
                    for price, ctx in hits[:6]:
                        w.writerow([town, price, ctx, url])
                    if hits:
                        print(f"    {len(hits)} candidate price(s) on {url[:60]}")
                        break  # first page with hits is usually the official schedule
                    time.sleep(0.8)
            except Exception as e:
                print(f"    error: {e}")
            time.sleep(1.2)  # be polite
    print(f"\nDone -> {args.out}. Upload this file back to the Claude conversation for verification + analysis.")

if __name__ == "__main__":
    main()
