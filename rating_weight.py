import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# ==============================================================================
# "MONEYBALL" RATING vs. STYLE WEIGHT FINDER
# ==============================================================================
# This script's purpose is to find the relative importance of Rating vs. Style,
# by intentionally IGNORING the 'h2h_advantage' feature.
# ==============================================================================

def create_game_features(games_df, features_df):
    """
    Creates feature differences for each game, focusing only on rating and style.
    """
    game_features = []
    
    for _, game in games_df.iterrows():
        p1_name, p2_name = game['winner_name'], game['loser_name']
        
        if p1_name not in features_df.index or p2_name not in features_df.index:
            continue
            
        p1_stats, p2_stats = features_df.loc[p1_name], features_df.loc[p2_name]
        
        # 1. Rating Difference
        rating_diff = game['winner_rating'] - game['loser_rating']
        
        # 2. Style Advantage (Combined)
        style_advantage = (p1_stats['Speed'] - p2_stats['Speed']) + \
                          (p1_stats['Consistency'] - p2_stats['Consistency']) + \
                          (p1_stats['Aggressiveness'] - p2_stats['Aggressiveness'])

        # Create symmetric samples for the model
        game_features.append({'rating_diff': rating_diff, 'style_advantage': style_advantage, 'winner': 1})
        game_features.append({'rating_diff': -rating_diff, 'style_advantage': -style_advantage, 'winner': 0})
        
    return pd.DataFrame(game_features)


def find_relative_weights(all_games_df, features_df):
    """
    Uses K-Fold Cross-Validation to find the relative weights of Rating and Style.
    """
    print("--- Step 1 of 3: Preparing data for Cross-Validation... ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_coeffs = []
    
    print("--- Step 2 of 3: Training model with 5-Fold Cross-Validation... ---")
    
    for fold, (train_index, test_index) in enumerate(kf.split(all_games_df)):
        print(f"  Training on Fold {fold+1}/5...")
        train_games = all_games_df.iloc[train_index]
        
        train_set = create_game_features(train_games, features_df)
        if train_set.empty: continue
        
        # --- THE KEY CHANGE IS HERE: We only use these two features ---
        features = ['rating_diff', 'style_advantage']
        X_train, y_train = train_set[features], train_set['winner']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        all_coeffs.append(np.abs(model.coef_[0]))
        
    avg_coeffs = np.mean(all_coeffs, axis=0)
    total_weight = np.sum(avg_coeffs)
    normalized_weights = (avg_coeffs / total_weight) * 100
    
    weights_df = pd.DataFrame({
        'Factor': ['Rating', 'Style'],
        'Learned Relative Weight (%)': normalized_weights
    })
    
    return weights_df.sort_values(by='Learned Relative Weight (%)', ascending=False)


# --- Main Execution Block ---
if __name__ == "__main__":
    
    try:
        player_features_df = pd.read_csv('player_features.csv', index_col='player_name')
        all_games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: 'player_features.csv' or 'all_games.csv' not found.")
        print("Please run the 'engineering.py' script first.")
        exit()
        
    relative_weights_df = find_relative_weights(all_games_df, player_features_df)
    
    print("--- Step 3 of 3: Presenting relative results... ---")
    print("\n\n===================================================================")
    print("            RELATIVE IMPORTANCE: RATING vs. STYLE")
    print("===================================================================")
    print("This model was trained WITHOUT H2H data to isolate the effects")
    print("of the two core factors.\n")
    
    print(relative_weights_df.round(1).to_string(index=False))
    
    print("\n--- Interpretation & Final Recommendation ---")
    print("This result tells you how much 'Style' matters beyond what raw 'Rating'")
    print("already explains. We can now combine this with our previous findings")
    print("to build the most intelligent and justifiable simulation weights.")
    print("===================================================================")