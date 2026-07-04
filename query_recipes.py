import sqlite3

# 1. Input ingredients (simulate CNN output)
detected_ingredients = ["apple","banana","beetroot","bell pepper","cabbage","capsicum","carrot",
"cauliflower","chilli pepper","corn","cucumber","eggplant","garlic","ginger",
"grapes","jalepeno","kiwi","lemon","lettuce","mango","onion","orange",
"paprika","pear","peas","pineapple","pomegranate","potato","raddish",
"soy beans","spinach","sweetcorn","sweetpotato","tomato","turnip","watermelon"]

# 2. Connect to database
conn = sqlite3.connect("nutriai.db")
cursor = conn.cursor()

# 3. Build placeholders for SQL IN clause
placeholders = ",".join("?" for _ in detected_ingredients)

# 4. SQL query: find recipes matching MOST ingredients
query = f"""
SELECT r.recipe_name, COUNT(DISTINCT i.ingredient_name) AS match_count
FROM recipes r
JOIN recipe_ingredients ri ON r.recipe_id = ri.recipe_id
JOIN ingredients i ON ri.ingredient_id = i.ingredient_id
WHERE i.ingredient_name IN ({placeholders})
GROUP BY r.recipe_id
ORDER BY match_count DESC;
"""

# 5. Execute query
cursor.execute(query, detected_ingredients)
results = cursor.fetchall()

conn.close()

# 6. Display results
print("Recommended recipes:")
for recipe, score in results:
    print(f"- {recipe} (matched {score} ingredients)")

