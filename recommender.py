# =============================================================================
# recommender_v2.py
# NutriAI v2 — Upgraded Recipe Recommender with Nutrition + User Preferences
# =============================================================================

import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "nutriai.db"

# =============================================================================
# HEALTH GOAL WEIGHTS
# Maps health_goal label → which nutrition field to favour
# =============================================================================
HEALTH_GOAL_CONFIG = {
    "Balanced":     {"tag": "balanced",     "field": None,        "low": False},
    "Low Sodium":   {"tag": "low_sodium",   "field": "sodium_mg", "low": True},
    "High Protein": {"tag": "high_protein", "field": "protein",   "low": False},
    "Low Calorie":  {"tag": "low_calorie",  "field": "calories",  "low": True},
    "Low Fat":      {"tag": "low_fat",      "field": "fat",       "low": True},
}

# =============================================================================
# TEXT NORMALISATION
# =============================================================================
INGREDIENT_SYNONYMS = {
    "aubergine": "eggplant",
    "brinjal": "eggplant",
    "capsicum": "bell pepper",
    "cilantro": "coriander",
    "chili": "chilli",
    "chili pepper": "chilli",
    "chilli pepper": "chilli",
    "green chili": "chilli",
    "green chilli": "chilli",
    "red chili": "chilli",
    "red chilli": "chilli",
    "scallion": "spring onion",
    "spring onions": "spring onion",
    "green onion": "spring onion",
    "green onions": "spring onion",
    "lady finger": "okra",
    "ladies finger": "okra",
    "ladys finger": "okra",
    "yardlong bean": "long bean",
    "yard long bean": "long bean",
    "long beans": "long bean",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "carrots": "carrot",
    "onions": "onion",
    "eggs": "egg",
}


def normalise(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in INGREDIENT_SYNONYMS:
        return INGREDIENT_SYNONYMS[text]

    # Simple singular fallback for English ingredient labels.
    if text.endswith("ies") and len(text) > 4:
        text = text[:-3] + "y"
    elif text.endswith("es") and len(text) > 4:
        text = text[:-2]
    elif text.endswith("s") and len(text) > 3:
        text = text[:-1]

    return INGREDIENT_SYNONYMS.get(text, text)


def normalise_set(items):
    return {normalise(i) for i in items if str(i).strip()}

# =============================================================================
# DATABASE LOADER
# =============================================================================
def load_recipes(db_path: str = DB_PATH):
    """
    Load all recipes with their ingredient lists from nutriai_v2.db.
    Returns a list of dicts.
    """
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT recipe_id, recipe_name, original_recipe_name, ingredients,
               cooking_steps, cooking_method, meal_tags, servings,
               calories, carbohydrate, protein, fat, sodium_mg,
               sodium_point, tips, source_book, health_tags, diet_type
        FROM recipes
    """)
    rows = cursor.fetchall()
    cols = [
        "recipe_id", "recipe_name", "original_recipe_name", "ingredients",
        "cooking_steps", "cooking_method", "meal_tags", "servings",
        "calories", "carbohydrate", "protein", "fat", "sodium_mg",
        "sodium_point", "tips", "source_book", "health_tags", "diet_type"
    ]

    recipes = []
    for row in rows:
        r = dict(zip(cols, row))
        # Parse ingredients into a list
        r["ingredient_list"] = [
            i.strip() for i in (r["ingredients"] or "").split(",")
            if i.strip()
        ]
        recipes.append(r)

    conn.close()
    return recipes

# =============================================================================
# NUTRITION SCORE
# =============================================================================
def nutrition_score(recipe: dict, health_goal: str) -> float:
    """
    Returns a score 0-1 based on how well the recipe matches the health goal.
    Uses health_tags stored in the database — no external calculation.
    """
    cfg = HEALTH_GOAL_CONFIG.get(health_goal, HEALTH_GOAL_CONFIG["Balanced"])
    tags = (recipe.get("health_tags") or "").lower()

    # Direct tag match gives full score
    if cfg["tag"] in tags:
        return 1.0

    # Partial: if no specific goal, all recipes get 0.5 (neutral)
    if cfg["field"] is None:
        return 0.5

    # Fallback: use numeric field relative scoring
    # (done at ranking time via normalisation, not here)
    return 0.3

# =============================================================================
# INGREDIENT MATCH SCORE
# =============================================================================
def ingredient_match_score(
    detected_norm: set,
    recipe_norm: set,
    detected_confidences: dict,
) -> tuple:
    """
    Returns (coverage, confidence, match_ratio, matched_set).
    """
    matched = detected_norm & recipe_norm
    if not matched:
        return 0.0, 0.0, 0.0, set()

    coverage    = len(matched) / len(recipe_norm) if recipe_norm else 0.0
    confidence  = sum(detected_confidences.get(i, 0.0) for i in matched) / len(matched)
    match_ratio = len(matched) / len(detected_norm) if detected_norm else 0.0

    return coverage, confidence, match_ratio, matched

# =============================================================================
# COMBINED SCORE
# =============================================================================
def combined_score(
    coverage: float,
    confidence: float,
    match_ratio: float,
    nutr_score: float,
) -> float:
    """
    Weighted combination:
        0.40 * coverage
        0.25 * confidence
        0.15 * match_ratio
        0.20 * nutrition_score
    """
    return (0.40 * coverage +
            0.25 * confidence +
            0.15 * match_ratio +
            0.20 * nutr_score)


def overlap_bonus(matched_count: int) -> float:
    """
    Small bonus for recipes matching multiple detected ingredients.
    Keeps ranking simple while favouring stronger recipe-ingredient overlap.
    """
    if matched_count <= 1:
        return 0.0
    return min((matched_count - 1) * 0.03, 0.09)

# =============================================================================
# EXPLANATION BUILDER
# =============================================================================
def build_reason(
    matched: set,
    coverage: float,
    health_goal: str,
    nutr_score: float,
    recipe: dict,
) -> str:
    matched_str = ", ".join(sorted(matched)) if matched else "none"
    parts = [
        f"Matched {len(matched)} ingredient(s): {matched_str}.",
        f"Ingredient coverage: {coverage:.0%}.",
    ]
    if nutr_score >= 1.0:
        parts.append(f"Meets your '{health_goal}' health goal.")
    elif nutr_score >= 0.5:
        parts.append(f"Partially suits '{health_goal}' goal.")
    else:
        parts.append(f"Does not specifically target '{health_goal}' goal.")
    return " ".join(parts)

# =============================================================================
# FILTERS
# =============================================================================
def passes_filters(
    recipe: dict,
    meal_filter: str,
    diet_filter: str,
    cooking_method_filter: str,
) -> bool:
    if meal_filter and meal_filter.lower() != "any":
        tags = (recipe.get("meal_tags") or "").lower()
        if meal_filter.lower() not in tags:
            return False

    if diet_filter and diet_filter.lower() != "any":
        dtype = (recipe.get("diet_type") or "").lower()
        if diet_filter.lower() not in dtype:
            return False

    if cooking_method_filter and cooking_method_filter.lower() != "any":
        method = (recipe.get("cooking_method") or "").lower()
        if cooking_method_filter.lower() not in method:
            return False

    return True

# =============================================================================
# MAIN RECOMMENDER
# =============================================================================
def recommend_recipes(
    detected_ingredients: list,
    detected_confidences: dict,
    health_goal: str          = "Balanced",
    meal_filter: str          = "Any",
    diet_filter: str          = "Any",
    cooking_method_filter: str = "Any",
    top_n: int                = 10,
    min_overlap: int          = 1,
    db_path: str              = DB_PATH,
) -> list:
    """
    Recommend recipes based on detected ingredients and user preferences.

    Parameters
    ----------
    detected_ingredients    : list[str]  — ingredient names from model
    detected_confidences    : dict[str, float]  — name → confidence
    health_goal             : str  — Balanced / Low Sodium / High Protein /
                                     Low Calorie / Low Fat
    meal_filter             : str  — Any / Breakfast / Lunch / Dinner /
                                     Snack / Beverage / Dessert
    diet_filter             : str  — Any / Vegetarian / Vegan / Egg /
                                     Chicken / Beef / Seafood / Mixed
    cooking_method_filter   : str  — Any / steam / boil / grill / bake /
                                     stir_fry / fry / roast / mix
    top_n                   : int  — number of results to return
    min_overlap             : int  — minimum matched ingredients required
    db_path                 : str  — path to SQLite database

    Returns
    -------
    list[dict]  sorted by combined_score descending
    """
    detected_norm = normalise_set(detected_ingredients)
    conf_norm     = {normalise(k): v for k, v in detected_confidences.items()}

    recipes = load_recipes(db_path)
    results = []

    for recipe in recipes:
        # Apply filters first (fast rejection)
        if not passes_filters(recipe, meal_filter, diet_filter,
                               cooking_method_filter):
            continue

        recipe_norm = normalise_set(recipe["ingredient_list"])
        if not recipe_norm:
            continue

        coverage, confidence, match_ratio, matched = ingredient_match_score(
            detected_norm, recipe_norm, conf_norm
        )

        if len(matched) < min_overlap:
            continue

        nutr_score = nutrition_score(recipe, health_goal)
        score      = combined_score(coverage, confidence, match_ratio, nutr_score)
        score      = min(score + overlap_bonus(len(matched)), 1.0)
        reason     = build_reason(matched, coverage, health_goal, nutr_score, recipe)

        results.append({
            "recipe_name":          recipe["recipe_name"],
            "original_recipe_name": recipe["original_recipe_name"] or "",
            "matched_ingredients":  sorted(matched),
            "match_score":          round(score, 4),
            "coverage":             round(coverage, 4),
            "confidence":           round(confidence, 4),
            "match_ratio":          round(match_ratio, 4),
            "nutrition_score":      round(nutr_score, 4),
            "calories":             recipe["calories"],
            "carbohydrate":         recipe["carbohydrate"],
            "protein":              recipe["protein"],
            "fat":                  recipe["fat"],
            "sodium_mg":            recipe["sodium_mg"],
            "sodium_point":         recipe["sodium_point"],
            "servings":             recipe["servings"],
            "cooking_method":       recipe["cooking_method"] or "",
            "meal_tags":            recipe["meal_tags"] or "",
            "diet_type":            recipe["diet_type"] or "",
            "health_tags":          recipe["health_tags"] or "",
            "tips":                 recipe["tips"] or "",
            "cooking_steps":        recipe["cooking_steps"] or "",
            "source_book":          recipe["source_book"] or "",
            "reason":               reason,
        })

    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results[:top_n]

# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == "__main__":
    sample_ingredients  = ["egg", "tomato", "onion", "garlic", "carrot"]
    sample_confidences  = {
        "egg":    0.92, "tomato": 0.85,
        "onion":  0.78, "garlic": 0.70, "carrot": 0.65,
    }

    print("\n[INFO] Detected:", sample_ingredients)
    results = recommend_recipes(
        detected_ingredients=sample_ingredients,
        detected_confidences=sample_confidences,
        health_goal="Low Sodium",
        meal_filter="Any",
        diet_filter="Any",
        cooking_method_filter="Any",
        top_n=5,
    )

    if not results:
        print("[INFO] No recipes found.")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n#{i} {r['recipe_name']} ({r['original_recipe_name']})")
            print(f"   Score  : {r['match_score']}")
            print(f"   Matched: {', '.join(r['matched_ingredients'])}")
            print(f"   Reason : {r['reason']}")
