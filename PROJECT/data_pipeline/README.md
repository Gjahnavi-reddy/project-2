# Data Pipeline

Pipeline: `books.toscrape.com` → scrape → clean → GBP/INR conversion → normalized SQLite → SQL + pandas verification.

The required fixed conversion is `1 GBP = 105.50 INR`; it is not a live exchange rate.

Run:

```bash
python scrape_pipeline.py
```

Outputs:
- `books_clean.csv`
- `books.db`
- `sql_results.txt`
