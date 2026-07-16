import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

def classify_ingredient(name):
    name = str(name).lower()
    
    # 1. Animal Protein (Meat, Fish, Poultry)
    meat_keywords = ['chicken', 'beef', 'steak', 'pork', 'bacon', 'turkey', 'duck', 'sausage', 
                     'ham', 'fish', 'shrimp', 'salmon', 'tuna', 'cod', 'prawn', 'meatball', 
                     'lamb', 'pork belly', 'prosciutto', 'salami', 'pepperoni']
    if any(kw in name for kw in meat_keywords):
        return 'Meat & Seafood'
        
    # 2. Eggs & Dairy
    dairy_keywords = ['cheese', 'cottage cheese', 'mozzarella', 'cheddar', 'feta', 'parmesan', 
                      'yogurt', 'milk', 'cream', 'egg', 'scrambled', 'omelet']
    if any(kw in name for kw in dairy_keywords):
        return 'Dairy & Eggs'
        
    # 3. Grains & Starches
    grain_keywords = ['rice', 'bread', 'pasta', 'noodle', 'spaghetti', 'macaroni', 'toast', 
                      'waffle', 'pancake', 'oatmeal', 'tortilla', 'pita', 'bagel', 'croissant',
                      'quinoa', 'millet', 'barley', 'corn', 'potato', 'hash brown', 'fries']
    if any(kw in name for kw in grain_keywords):
        return 'Grains & Starches'
        
    # 4. Vegetables & Fruits
    veg_keywords = ['salad', 'green', 'lettuce', 'spinach', 'kale', 'romaine', 'cabbage', 'bok choy',
                    'broccoli', 'cauliflower', 'asparagus', 'brussels sprout', 'squash', 'zucchini',
                    'cucumber', 'tomato', 'pepper', 'carrot', 'mushroom', 'onion', 'garlic', 
                    'celery', 'bean sprout', 'peas', 'avocado', 'guacamole', 'olive']
    if any(kw in name for kw in veg_keywords):
        return 'Vegetables & Fruits'
        
    # 5. Fruits (excluding savory ones like avocado/olive)
    fruit_keywords = ['strawberry', 'berry', 'berries', 'apple', 'banana', 'kiwi', 'lemon', 'orange', 
                      'pineapple', 'cantaloupe', 'melon', 'grape', 'mango', 'peach', 'pear', 'lime']
    if any(kw in name for kw in fruit_keywords):
        return 'Vegetables & Fruits'
        
    # 6. Legumes, Nuts & Seeds
    nut_keywords = ['bean', 'lentil', 'chickpea', 'tofu', 'edamame', 'nut', 'almond', 'peanut', 'seed', 'sesame']
    if any(kw in name for kw in nut_keywords):
        return 'Legumes, Nuts & Soy'
        
    return 'Sauces, Oils & Condiments'

def detect_cuisine_profile(ingredients):
    # Safe checks for ingredients list
    if not isinstance(ingredients, list) or len(ingredients) == 0:
        return 'Mixed Cafeteria Plate'
        
    ingr_str = " ".join([str(i).lower() for i in ingredients])
    
    # 1. Single-Ingredient / Simple Dishes
    if len(ingredients) == 1:
        ingr = str(ingredients[0]).lower()
        fruit_keywords = ['strawberry', 'berry', 'berries', 'apple', 'banana', 'kiwi', 'lemon', 'orange', 
                          'pineapple', 'cantaloupe', 'melon', 'grape', 'mango', 'peach', 'pear', 'figs']
        if any(kw in ingr for kw in fruit_keywords):
            return 'Fruit Plates / Snacks'
        return 'Simple Sides / Single Items'
        
    # 2. Regional Cuisines Scores
    asian_kw = ['soy sauce', 'ginger', 'bok choy', 'sesame', 'tofu', 'millet', 'noodle', 'ramen', 
                'white rice', 'brown rice', 'spring roll', 'egg roll', 'crawfish', 'octopus']
    mexican_kw = ['tortilla', 'guacamole', 'salsa', 'taco', 'avocado', 'cilantro', 'black beans', 
                  'jalapeno', 'lime juice', 'refried beans']
    italian_kw = ['pasta', 'spaghetti', 'lasagna', 'mozzarella', 'parmesan', 'basil', 'oregano', 
                  'caesar salad', 'pasta salad', 'olive oil', 'pesto', 'macaroni and cheese']
    western_kw = ['bacon', 'hash brown', 'steak', 'fries', 'burger', 'hot dog', 'pizza', 'sausage', 
                  'ham', 'waffle', 'pancake', 'brisket', 'potato chips', 'ketchup', 'mayonnaise', 
                  'barbecue sauce']
    
    asian_score = sum(1 for kw in asian_kw if kw in ingr_str)
    mexican_score = sum(1 for kw in mexican_kw if kw in ingr_str)
    italian_score = sum(1 for kw in italian_kw if kw in ingr_str)
    western_score = sum(1 for kw in western_kw if kw in ingr_str)
    
    scores = {
        'Asian': asian_score,
        'Mexican': mexican_score,
        'Italian / Mediterranean': italian_score,
        'Western / American': western_score
    }
    
    max_score = max(scores.values())
    if max_score > 0:
        best_cuisines = [k for k, v in scores.items() if v == max_score]
        return best_cuisines[0]
        
    # 3. Fallback Subcategories (Splitting the generic Cafeteria Mix)
    
    # Breakfast & Bakery
    breakfast_kw = ['egg', 'scrambled', 'poached', 'omelet', 'oatmeal', 'pancake', 'waffle', 
                    'syrup', 'croissant', 'pastries', 'muffin', 'toast', 'biscuit', 'raisin bran', 
                    'cereal', 'oats', 'bagel']
    if any(kw in ingr_str for kw in breakfast_kw):
        return 'Breakfast & Bakery'
        
    # Seafood & Fish Plates
    seafood_kw = ['fish', 'shrimp', 'salmon', 'tuna', 'cod', 'octopus', 'clams', 'mussels', 
                  'crawfish', 'seafood', 'tilapia', 'herring', 'crab', 'lobster']
    if any(kw in ingr_str for kw in seafood_kw):
        return 'Seafood & Fish Plates'
        
    # Salad Bar / Greens
    if any(kw in ingr_str for kw in ['salad', 'greens', 'lettuce', 'spinach', 'kale', 'romaine', 'cabbage']):
        return 'Salad Bar / Greens'
        
    # Healthy Grains & Legumes
    healthy_kw = ['quinoa', 'wild rice', 'brown rice', 'chickpea', 'lentil', 'edamame', 'falafel', 
                  'tabouli', 'beans', 'bean(seed)', 'mixed nuts', 'almonds', 'pecans', 'walnuts']
    if any(kw in ingr_str for kw in healthy_kw):
        return 'Healthy Grains & Legumes'
        
    return 'Mixed Cafeteria Plate'

def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_dir, 'AI4All Dataset')
    plots_dir = os.path.join(dataset_dir, 'plots')
    
    os.makedirs(plots_dir, exist_ok=True)
    
    dish_ingredients_path = os.path.join(dataset_dir, 'dish_ingredients.csv')
    
    if not os.path.exists(dish_ingredients_path):
        print(f"[-] Error: Ingredients CSV file not found at: {dish_ingredients_path}")
        return
        
    print("[+] Loading dish ingredients mapping...")
    df_ingr = pd.read_csv(dish_ingredients_path)
    
    # 1. Map each ingredient row to its general food group
    print("[+] Categorizing ingredients into food groups...")
    df_ingr['food_group'] = df_ingr['ingr_name'].apply(classify_ingredient)
    
    # For each dish, determine which food groups are present (binary presence)
    dish_groups = df_ingr.groupby(['dish_id', 'food_group']).size().unstack(fill_value=0)
    dish_groups_binary = (dish_groups > 0).astype(int)
    
    # Count how many dishes contain each food group
    food_group_counts = dish_groups_binary.sum().sort_values(ascending=False)
    food_group_pct = (food_group_counts / len(dish_groups_binary)) * 100
    
    print("\n" + "="*65)
    print("      FOOD GROUPS PRESENCE ACROSS DISHES (N = 4,768)")
    print("="*65)
    for group, count in food_group_counts.items():
        pct = food_group_pct[group]
        print(f" * {group:<28} : {count:>5} dishes ({pct:>5.2f}%)")
    print("="*65 + "\n")
    
    # Plot Food Group Distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=food_group_pct.values, y=food_group_pct.index, palette='crest')
    plt.title('Percentage of Dishes Containing Each Food Group')
    plt.xlabel('Percentage of Dishes (%)')
    plt.ylabel('Food Group')
    plt.xlim(0, 100)
    for i, v in enumerate(food_group_pct.values):
        plt.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='semibold')
    
    food_types_plot = os.path.join(plots_dir, 'food_types_distribution.png')
    plt.savefig(food_types_plot, dpi=300, bbox_inches='tight')
    print(f"[+] Saved food groups distribution plot to: {food_types_plot}")
    
    # 2. Cuisine Profiling based on ingredient presence
    print("[+] Profiling cuisines and categorizing cafeteria mix...")
    dish_ingredients_list = df_ingr.groupby('dish_id')['ingr_name'].apply(list).reset_index()
    dish_ingredients_list['cuisine_profile'] = dish_ingredients_list['ingr_name'].apply(detect_cuisine_profile)
    
    cuisine_counts = dish_ingredients_list['cuisine_profile'].value_counts()
    cuisine_pct = dish_ingredients_list['cuisine_profile'].value_counts(normalize=True) * 100
    
    print("\n" + "="*65)
    print("          ESTIMATED CUISINE/FOOD TYPE PROFILES")
    print("="*65)
    for cuisine, count in cuisine_counts.items():
        pct = cuisine_pct[cuisine]
        print(f" * {cuisine:<28} : {count:>5} dishes ({pct:>5.2f}%)")
    print("="*65 + "\n")
    
    # Plot Cuisine Profile Distribution
    plt.figure(figsize=(12, 7))
    sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, palette='Set2')
    plt.title('Distribution of Dishes by Cuisine/Food Profile\n(Cafeteria Mix split into subcategories)')
    plt.xlabel('Number of Dishes')
    plt.ylabel('Cuisine / Food Profile')
    for i, v in enumerate(cuisine_counts.values):
        plt.text(v + 10, i, f"{v} ({cuisine_pct.iloc[i]:.1f}%)", va='center')
        
    cuisine_plot = os.path.join(plots_dir, 'food_cuisine_distribution.png')
    plt.savefig(cuisine_plot, dpi=300, bbox_inches='tight')
    print(f"[+] Saved cuisine profile distribution plot to: {cuisine_plot}")
    
    print("\n[+] Done! Cuisines and food types analyzed successfully.")

if __name__ == '__main__':
    main()
