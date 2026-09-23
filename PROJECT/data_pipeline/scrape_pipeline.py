from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com"
RATE_GBP_TO_INR = 105.50
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "books.db"
CSV_PATH = ROOT / "books_clean.csv"
RESULTS_PATH = ROOT / "sql_results.txt"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def scrape_pages(pages: int = 5) -> pd.DataFrame:
    rows = []
    for page in range(1, pages + 1):
        url = f"{BASE}/catalogue/page-{page}.html"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for card in soup.select("article.product_pod"):
            title = card.h3.a.get("title", "").strip()
            price_text = card.select_one(".price_color").get_text(strip=True)
            availability = card.select_one(".availability").get_text(" ", strip=True)
            rating_class = next(
                (c for c in card.select_one(".star-rating").get("class", [])
                 if c in RATING_MAP),
                None,
            )
            category_url = card.select_one("h3 a")["href"]
            # The category is not present on the product card. The catalogue page
            # therefore uses the breadcrumb/category page when available.
            detail_url = requests.compat.urljoin(url, category_url)
            detail = requests.get(detail_url, timeout=20)
            detail.raise_for_status()
            detail_soup = BeautifulSoup(detail.text, "html.parser")
            crumbs = detail_soup.select("ul.breadcrumb li a")
            category = crumbs[-1].get_text(strip=True) if crumbs else "Unknown"

            rows.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": rating_class or "Unknown",
                    "availability": availability,
                    "category": category,
                }
            )
    return pd.DataFrame(rows)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_gbp"] = (
        out["price"].astype(str).str.replace("£", "", regex=False).str.strip()
    )
    out["price_gbp"] = pd.to_numeric(out["price_gbp"], errors="coerce")
    out["rating"] = out["star_rating"].map(RATING_MAP)
    out["in_stock"] = out["availability"].str.contains(
        "In stock", case=False, na=False
    )
    out["price_gbp"] = out["price_gbp"].fillna(out["price_gbp"].median())
    out = out.dropna(subset=["rating", "category", "title"]).copy()
    out["rating"] = out["rating"].astype(int)
    out["price_inr"] = (out["price_gbp"] * RATE_GBP_TO_INR).round(2)
    return out[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ]


def create_database(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            );
            """
        )

        categories = (
            df[["category"]].drop_duplicates().sort_values("category").reset_index(drop=True)
        )
        categories.insert(0, "category_id", range(1, len(categories) + 1))
        categories.to_sql("categories", conn, if_exists="append", index=False)

        books = df.merge(categories, on="category", how="left")
        books.insert(0, "book_id", range(1, len(books) + 1))
        books["in_stock"] = books["in_stock"].astype(int)
        books = books[
            ["book_id", "title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
        ]
        books.to_sql("books", conn, if_exists="append", index=False)


def run_sql_examples() -> None:
    queries = {
        "1_SELECT_WHERE": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
            LIMIT 10;
        """,
        "2_ORDER_BY_LIMIT": """
            SELECT title, price_inr
            FROM books
            ORDER BY price_inr DESC
            LIMIT 10;
        """,
        "3_DISTINCT": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating;
        """,
        "4_BETWEEN": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp;
        """,
        "5_JOIN": """
            SELECT c.category_name, b.title, b.rating, b.price_inr
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY c.category_name, b.rating DESC, b.title
            LIMIT 10;
        """,
    }

    with sqlite3.connect(DB_PATH) as conn, RESULTS_PATH.open("w", encoding="utf-8") as f:
        for name, query in queries.items():
            f.write(f"\n=== {name} ===\n{query.strip()}\n")
            result = pd.read_sql(query, conn)
            f.write(result.to_string(index=False) + "\n")

        join_sql = pd.read_sql(queries["5_JOIN"], conn)

        books = pd.read_sql("SELECT * FROM books", conn)
        categories = pd.read_sql("SELECT * FROM categories", conn)
        join_pandas = (
            books.merge(categories, on="category_id", how="inner")
            [["category_name", "title", "rating", "price_inr"]]
            .sort_values(["category_name", "rating", "title"], ascending=[True, False, True])
            .head(10)
            .reset_index(drop=True)
        )

        f.write("\n=== pandas JOIN EQUIVALENCE ===\n")
        f.write(f"Equivalent: {join_sql.reset_index(drop=True).equals(join_pandas)}\n")
        f.write("\nSQL result:\n" + join_sql.to_string(index=False) + "\n")
        f.write("\npandas.merge result:\n" + join_pandas.to_string(index=False) + "\n")


def main() -> None:
    raw = scrape_pages(pages=5)
    clean = clean_data(raw)
    if len(clean) < 60:
        raise RuntimeError(f"Expected at least 60 rows, got {len(clean)}")
    if clean["category"].nunique() < 3:
        raise RuntimeError("Expected at least 3 categories.")
    clean.to_csv(CSV_PATH, index=False)
    create_database(clean)
    run_sql_examples()
    print(f"Completed: {len(clean)} books across {clean['category'].nunique()} categories.")
    print(f"Database: {DB_PATH}")
    print(f"SQL results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
