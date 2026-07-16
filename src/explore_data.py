import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

def main():
    # Define paths based on your actual workspace directory structure
    project_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(project_dir, 'AI4All Dataset')
    plots_dir = os.path.join(dataset_dir, 'plots')
    
    os.makedirs(plots_dir, exist_ok=True)
    
    dish_nutrition_path = os.path.join(dataset_dir, 'dish_nutrition_values.csv')
    dish_ingredients_path = os.path.join(dataset_dir, 'dish_ingredients.csv')
    
    # Check if essential files exist
    if not os.path.exists(dish_nutrition_path):
        print("\n" + "="*80)
        print("[-] ERROR: nutrition values CSV file not found!")
        print(f"Expected file at: {dish_nutrition_path}")
        print("="*80 + "\n")
        return

    print("[+] Loading dataset files...")
    df_nutrition = pd.read_csv(dish_nutrition_path)
    
    if os.path.exists(dish_ingredients_path):
        print("[+] Loading dish ingredients details...")
        df_ingredients = pd.read_csv(dish_ingredients_path)
        ingr_counts = df_ingredients.groupby('dish_id').size().reset_index(name='num_ingredients')
        df = pd.merge(df_nutrition, ingr_counts, on='dish_id', how='left')
        df['num_ingredients'] = df['num_ingredients'].fillna(0).astype(int)
    else:
        df = df_nutrition

    print(f"\n[+] Successfully loaded {len(df)} total food dishes!")
    
    # Categorize dishes into Low, Medium, High calorie ranges
    # Low: < 150 kcal, Medium: 150 - 400 kcal, High: >= 400 kcal
    def get_calorie_class(cal):
        if cal < 150:
            return 'Low Calorie (<150 kcal)'
        elif cal < 400:
            return 'Medium Calorie (150-400 kcal)'
        else:
            return 'High Calorie (>=400 kcal)'
            
    df['calorie_category'] = df['calories'].apply(get_calorie_class)
    category_counts = df['calorie_category'].value_counts()
    category_pct = df['calorie_category'].value_counts(normalize=True) * 100

    print("\n" + "="*65)
    print("           DISH CALORIE CATEGORY DISTRIBUTION")
    print("="*65)
    for cat in ['Low Calorie (<150 kcal)', 'Medium Calorie (150-400 kcal)', 'High Calorie (>=400 kcal)']:
        count = category_counts.get(cat, 0)
        pct = category_pct.get(cat, 0.0)
        print(f" * {cat:<32} : {count:>5} dishes ({pct:>5.2f}%)")
    print("="*65 + "\n")

    # Plot Category Countplot
    plt.figure(figsize=(10, 6))
    category_order = ['Low Calorie (<150 kcal)', 'Medium Calorie (150-400 kcal)', 'High Calorie (>=400 kcal)']
    sns.countplot(data=df, x='calorie_category', order=category_order, palette='viridis')
    plt.title('Dish Calorie Class Distribution')
    plt.xlabel('Calorie Category')
    plt.ylabel('Number of Dishes')
    # Add count values on top of bars
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')
    category_plot_path = os.path.join(plots_dir, 'dish_category_distribution.png')
    plt.savefig(category_plot_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved calorie category distribution plot to: {category_plot_path}")

    # For distributions of continuous values, we filter out extreme outliers (top 1.5% values)
    # to make the visual plots readable.
    cutoff_q = 0.985
    
    # 1. Enhanced Calorie Distribution
    plt.figure()
    cal_cutoff = df['calories'].quantile(cutoff_q)
    df_filtered_cal = df[df['calories'] <= cal_cutoff]
    sns.histplot(data=df_filtered_cal, x='calories', kde=True, color='crimson', bins=35)
    plt.title(f'Enhanced Calorie Distribution (Excluding Top 1.5% Outliers > {int(cal_cutoff)} kcal)')
    plt.xlabel('Calories (kcal)')
    plt.ylabel('Count')
    plt.xlim(0, cal_cutoff)
    calorie_plot_path = os.path.join(plots_dir, 'calorie_distribution.png')
    plt.savefig(calorie_plot_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved enhanced calorie distribution plot to: {calorie_plot_path}")
    
    # 2. Enhanced Mass Distribution
    plt.figure()
    mass_cutoff = df['mass'].quantile(cutoff_q)
    df_filtered_mass = df[df['mass'] <= mass_cutoff]
    sns.histplot(data=df_filtered_mass, x='mass', kde=True, color='teal', bins=35)
    plt.title(f'Enhanced Food Mass Distribution (Excluding Top 1.5% Outliers > {int(mass_cutoff)}g)')
    plt.xlabel('Mass (g)')
    plt.ylabel('Count')
    plt.xlim(0, mass_cutoff)
    mass_plot_path = os.path.join(plots_dir, 'mass_distribution.png')
    plt.savefig(mass_plot_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved enhanced mass distribution plot to: {mass_plot_path}")
    
    # 3. Enhanced Macronutrients Distribution (Fat, Carbs, Protein)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    fat_cutoff = df['fat'].quantile(cutoff_q)
    sns.histplot(data=df[df['fat'] <= fat_cutoff], x='fat', kde=True, color='orange', ax=axes[0], bins=30)
    axes[0].set_title(f'Fat Distribution (Up to {int(fat_cutoff)}g)')
    axes[0].set_xlabel('Fat (g)')
    axes[0].set_xlim(0, fat_cutoff)
    
    carb_cutoff = df['carb'].quantile(cutoff_q)
    sns.histplot(data=df[df['carb'] <= carb_cutoff], x='carb', kde=True, color='purple', ax=axes[1], bins=30)
    axes[1].set_title(f'Carbs Distribution (Up to {int(carb_cutoff)}g)')
    axes[1].set_xlabel('Carbs (g)')
    axes[1].set_xlim(0, carb_cutoff)
    
    protein_cutoff = df['protein'].quantile(cutoff_q)
    sns.histplot(data=df[df['protein'] <= protein_cutoff], x='protein', kde=True, color='forestgreen', ax=axes[2], bins=30)
    axes[2].set_title(f'Protein Distribution (Up to {int(protein_cutoff)}g)')
    axes[2].set_xlabel('Protein (g)')
    axes[2].set_xlim(0, protein_cutoff)
    
    plt.suptitle('Enhanced Macronutrient Distributions (Excluding Top 1.5% Outliers)', y=1.02, fontsize=16)
    macro_plot_path = os.path.join(plots_dir, 'macronutrient_distribution.png')
    plt.savefig(macro_plot_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved enhanced macronutrients distribution plot to: {macro_plot_path}")
    
    # 4. Enhanced Ingredient count distribution if available
    if 'num_ingredients' in df.columns:
        plt.figure(figsize=(10, 6))
        ingr_cutoff = int(df['num_ingredients'].quantile(cutoff_q))
        df_filtered_ingr = df[df['num_ingredients'] <= ingr_cutoff]
        sns.countplot(data=df_filtered_ingr, x='num_ingredients', color='royalblue')
        plt.title(f'Number of Ingredients per Dish (Up to {ingr_cutoff} ingredients)')
        plt.xlabel('Number of Ingredients')
        plt.ylabel('Count')
        ingr_plot_path = os.path.join(plots_dir, 'ingredients_distribution.png')
        plt.savefig(ingr_plot_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved enhanced ingredient count distribution plot to: {ingr_plot_path}")
        
    print("\n[+] Done! All enhanced distributions calculated and plotted successfully.")

if __name__ == '__main__':
    main()
