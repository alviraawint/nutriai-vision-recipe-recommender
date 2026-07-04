import csv
import sqlite3

conn = sqlite3.connect("nutriai.db")
cursor = conn.cursor()

with open("recipes.csv", newline="", encoding="utf-8-sig") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        recipe_name = row["recipe_name"]
        ingredients = row["ingredients"]
        cooking_steps = row["cooking_steps"]

        cursor.execute("""
        INSERT INTO recipes (recipe_name, cooking_steps)
        VALUES (?, ?)
        """, (recipe_name, cooking_steps))

        cursor.execute(
            "SELECT recipe_id FROM recipes WHERE recipe_name = ?",
            (recipe_name,)
        )
        recipe_id = cursor.fetchone()[0]

        ingredient_list = [i.strip() for i in ingredients.split(",")]

        for ing in ingredient_list:
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
                "INSERT INTO recipe_ingredients VALUES (?, ?)",
                (recipe_id, ingredient_id)
            )

conn.commit()
conn.close()

print("✅ CSV data imported into SQL successfully!")

import sqlite3

conn = sqlite3.connect("nutriai.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM recipes")
print(cursor.fetchone())

conn.close()
