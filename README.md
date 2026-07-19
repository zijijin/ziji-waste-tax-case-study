# zj-waste-tax-case-study
# Waste Tax Case Study — Research Toolkit

Supporting code and data for my case study response: *"If Waste Had a Price Tag: How Much Less Would We Actually Throw Away?"* (Scenario 3: a policy accurately measures each person's waste production and taxes them on it).

Everything here was built during the 48-hour case window. AI tools were used throughout and are documented in the GenAI appendix; in short, Claude wrote code with me and ran the analysis, GPT served as an adversarial reviewer, and Perplexity verified external sources.

## What's in this repo

### `/pipeline` — Massachusetts panel builder
- `etl_massdep.py` — reads five years of MassDEP's municipal solid-waste survey Excel files (2021–2025, all 351 municipalities), locates the key columns by header name across differently-formatted years, applies quality screens, and outputs a clean town-by-year panel (1,306 usable observations).
- `analysis.py` — merges Census ACS demographics (pulled via the free Census API) and land area, then runs the comparisons in the paper: the pooled regression with demographic controls, nearest-neighbor matching, the alternative-denominator check, and the within-town switcher analysis. Standard errors clustered by municipality.

### `/scraper` — bag-price harvester
- `payt_price_scraper.py` — for each pay-as-you-throw town, searches the web for the town's official trash page, ranks municipal domains above blogs and news, downloads candidate pages, and extracts every dollar amount appearing near bag-related keywords, saving the surrounding sentence and source URL for each hit. Deliberately human-in-the-loop: it outputs *candidates*, which I then verified by hand against each source page (that verification caught real errors, including an aggregator page describing Worcester's program under a different town's name).
- `payt_towns.csv` — the 124 target towns, from my panel.
- `payt_price_candidates.csv` — the raw scraper output (109 candidates across 33 towns).


### `/basket` — retail price basket
- `basket_completed_42.csv` — 42 products in matched pairs (wasteful format vs. low-waste format of the same brand), with prices and source URLs captured July 17–18, 2026.
- `basket_analysis.py`, `basket_results.csv`, `basket_breakeven.csv` — per-use cost comparison and break-even waste-tax calculation for each pair. Headline: in 14 of 15 clean pairs the low-waste option is already cheaper per use.

### `/docs`
- `evidence_ledger.csv` — every claim in the paper with its source, calculation, confidence level, and open objections. This was the working spine of the project.

## How to reproduce
```bash
pip install pandas openpyxl statsmodels scipy matplotlib requests beautifulsoup4 ddgs
python pipeline/etl_massdep.py      # needs the five MassDEP survey files (links in the paper's appendix)
python pipeline/analysis.py
python scraper/payt_price_scraper.py --towns scraper/payt_towns.csv --limit 40
python basket/basket_analysis.py
```

## Data sources
MassDEP Municipal Solid Waste & Recycling Survey (CY2021–2025); U.S. Census Bureau ACS 2023 5-year (via API); municipal websites (fee schedules); retailer product pages (basket prices); full citation list in the paper's Appendix A.
