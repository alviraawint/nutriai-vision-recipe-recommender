# =============================================================================
# create_database.py
# NutriAI v2 — Build SQLite database from recipes_master.csv
# =============================================================================

import re
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "recipes.csv"
DB_PATH = BASE_DIR / "nutriai.db"

REQUIRED_COLUMNS = {
    "recipe_name",
    "ingredients",
    "ingredients_original",
    "cooking_steps",
    "cooking_method",
    "meal_tags",
    "servings",
    "calories",
    "carbohydrate",
    "protein",
    "fat",
    "sodium_mg",
    "tips",
    "source_book",
    "health_tags",
    "diet_type",
}

# =============================================================================
# HELPERS
# =============================================================================
def safe_str(val):
    """Return stripped string or empty string for NaN."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def safe_float(val):
    """Return float or None for NaN/invalid."""
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    """Return int or None for NaN/invalid."""
    try:
        if pd.isna(val):
            return None
        text = str(val).strip()
        match = re.search(r"\d+", text)
        if match:
            return int(match.group(0))
        return int(float(text))
    except (ValueError, TypeError):
        return None


def validate_columns(df):
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "recipes.csv is missing required column(s): "
            + ", ".join(missing)
        )


def recreate_tables(cursor):
    cursor.execute("DROP TABLE IF EXISTS recipe_ingredients")
    cursor.execute("DROP TABLE IF EXISTS ingredients")
    cursor.execute("DROP TABLE IF EXISTS recipes")

# =============================================================================
# MAIN
# =============================================================================
def build_database():
    print(f"[INFO] Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    validate_columns(df)
    print(f"[INFO] Loaded {len(df)} rows.")

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    recreate_tables(cursor)

    # -------------------------------------------------------------------------
    # TABLE: recipes
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_name           TEXT NOT NULL,
            original_recipe_name  TEXT,
            ingredients           TEXT,
            ingredient_count      INTEGER,
            ingredients_original  TEXT,
            cooking_steps         TEXT,
            cooking_method        TEXT,
            meal_tags             TEXT,
            servings              INTEGER,
            calories              REAL,
            carbohydrate          REAL,
            protein               REAL,
            fat                   REAL,
            sodium_mg             REAL,
            sodium_point          REAL,
            tips                  TEXT,
            source_book           TEXT,
            health_tags           TEXT,
            diet_type             TEXT,
            UNIQUE(recipe_name)
        )
    """)

    # -------------------------------------------------------------------------
    # TABLE: ingredients  (normalized)
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE ingredients (
            ingredient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            ingredient_name TEXT UNIQUE NOT NULL
        )
    """)

    # -------------------------------------------------------------------------
    # TABLE: recipe_ingredients  (join table)
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE recipe_ingredients (
            recipe_id     INTEGER NOT NULL,
            ingredient_id INTEGER NOT NULL,
            FOREIGN KEY(recipe_id)     REFERENCES recipes(recipe_id),
            FOREIGN KEY(ingredient_id) REFERENCES ingredients(ingredient_id),
            PRIMARY KEY(recipe_id, ingredient_id)
        )
    """)

    conn.commit()

    # -------------------------------------------------------------------------
    # INSERT ROWS
    # -------------------------------------------------------------------------
    inserted = 0
    skipped  = 0

    for _, row in df.iterrows():
        recipe_name = safe_str(row.get("recipe_name", ""))
        if not recipe_name:
            skipped += 1
            continue

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO recipes (
                    recipe_name, original_recipe_name, ingredients,
                    ingredient_count, ingredients_original, cooking_steps,
                    cooking_method, meal_tags, servings, calories,
                    carbohydrate, protein, fat, sodium_mg, sodium_point,
                    tips, source_book, health_tags, diet_type
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                recipe_name,
                safe_str(row.get("original_recipe_name", "")),
                safe_str(row.get("ingredients", "")),
                safe_int(row.get("ingredient_count")),
                safe_str(row.get("ingredients_original", "")),
                safe_str(row.get("cooking_steps", "")),
                safe_str(row.get("cooking_method", "")),
                safe_str(row.get("meal_tags", "")),
                safe_int(row.get("servings")),
                safe_float(row.get("calories")),
                safe_float(row.get("carbohydrate")),
                safe_float(row.get("protein")),
                safe_float(row.get("fat")),
                safe_float(row.get("sodium_mg")),
                safe_float(row.get("sodium_point")),
                safe_str(row.get("tips", "")),
                safe_str(row.get("source_book", "")),
                safe_str(row.get("health_tags", "")),
                safe_str(row.get("diet_type", "")),
            ))

            # Only link ingredients for newly inserted recipes
            if cursor.rowcount == 1:
                recipe_id = cursor.lastrowid
                raw_ingredients = safe_str(row.get("ingredients", ""))

                for ing in raw_ingredients.split(","):
                    ing = ing.strip().lower()
                    if not ing:
                        continue

                    cursor.execute(
                        "INSERT OR IGNORE INTO ingredients (ingredient_name) VALUES (?)",
                        (ing,)
                    )
                    cursor.execute(
                        "SELECT ingredient_id FROM ingredients WHERE ingredient_name = ?",
                        (ing,)
                    )
                    ingredient_id = cursor.fetchone()[0]

                    cursor.execute(
                        "INSERT OR IGNORE INTO recipe_ingredients "
                        "(recipe_id, ingredient_id) VALUES (?,?)",
                        (recipe_id, ingredient_id)
                    )
                inserted += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"[WARN] Skipping row '{recipe_name}': {e}")
            skipped += 1

    conn.commit()
    conn.close()

    print(f"[DONE] Database saved to: {DB_PATH}")
    print(f"       Inserted : {inserted}")
    print(f"       Skipped  : {skipped}")


if __name__ == "__main__":
    build_database()
