import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# ==============================================================================
# "MONEYBALL" WEIGHT FINDER SCRIPT (ROBUST VERSION)
# ==============================================================================

def create_game_features(games_df, features_df, h2h_lookup):
    """
    Creates feature differences for each game, now using a pre-computed H2H lookup
    to prevent data leakage during cross-validation.
    """
    game_features = []
    
    for _, game in games_df.iterrows():
        p1_name, p2_name = game['winner_name'], game['loser_name']
        
        if p1_name not in features_df.index or p2_name not in features_df.index:
            continue
            
        p1_stats, p2_stats = features_df.loc[p1_name], features_df.loc[p2_name]
        
        # 1. Rating Difference
        rating_diff = game['winner_rating'] - game['loser_rating']
        
        # 2. H2H Advantage (from the pre-computed, leak-free lookup)
        h2h_advantage = h2h_lookup.get(p1_name, {}).get(p2_name, 0.5) # Default to 50% if no history
        
        # 3. Style Advantage
        style_advantage = (p1_stats['Speed'] - p2_stats['Speed']) + \
                          (p1_stats['Consistency'] - p2_stats['Consistency']) + \
                          (p1_stats['Aggressiveness'] - p2_stats['Aggressiveness'])

        # Create features for P1 (winner) vs P2 (loser)
        game_features.append({
            'rating_diff': rating_diff, 'h2h_advantage': h2h_advantage - 0.5,
            'style_advantage': style_advantage, 'winner': 1
        })
        # Create features for P2 (loser) vs P1 (winner)
        game_features.append({
            'rating_diff': -rating_diff, 'h2h_advantage': -(h2h_advantage - 0.5),
            'style_advantage': -style_advantage, 'winner': 0
        })
        
    return pd.DataFrame(game_features)


def find_robust_weights(all_games_df, features_df):
    """
    Uses K-Fold Cross-Validation to find robust, data-leak-free feature weights.
    """
    print("--- Step 1 of 3: Preparing data for Cross-Validation... ---")
    
    # KFold will split our data into 5 chunks (folds) for robust training
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_coeffs = []
    
    print("--- Step 2 of 3: Training model with 5-Fold Cross-Validation... ---")
    
    # Loop through each split
    for fold, (train_index, test_index) in enumerate(kf.split(all_games_df)):
        print(f"  Training on Fold {fold+1}/5...")
        
        # Define which games are for training and which are for testing in this fold
        train_games, test_games = all_games_df.iloc[train_index], all_games_df.iloc[test_index]
        
        # --- CRITICAL STEP: Prevent Data Leakage ---
        # Build the H2H lookup using ONLY the training games for this fold.
        h2h_lookup = {}
        h2h_groups = train_games.groupby(['winner_name', 'loser_name']).size().reset_index(name='wins')
        all_players = pd.concat([train_games['winner_name'], train_games['loser_name']]).unique()
        for p1 in all_players:
            h2h_lookup[p1] = {}
            for p2 in all_players:
                if p1 == p2: continue
                wins = h2h_groups[(h2h_groups['winner_name'] == p1) & (h2h_groups['loser_name'] == p2)]['wins'].sum()
                losses = h2h_groups[(h2h_groups['winner_name'] == p2) & (h2h_groups['loser_name'] == p1)]['wins'].sum()
                if (wins + losses) > 0:
                    h2h_lookup[p1][p2] = wins / (wins + losses)

        # Create the training and testing sets using this leak-free H2H lookup
        train_set = create_game_features(train_games, features_df, h2h_lookup)
        test_set = create_game_features(test_games, features_df, h2h_lookup)
        
        if train_set.empty or test_set.empty: continue
        
        # Prepare data for the model
        features = ['rating_diff', 'h2h_advantage', 'style_advantage']
        X_train, y_train = train_set[features], train_set['winner']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Store the learned coefficients from this fold
        all_coeffs.append(np.abs(model.coef_[0]))
        
    # Average the coefficients from all 5 folds to get a stable result
    avg_coeffs = np.mean(all_coeffs, axis=0)
    
    # Normalize the final average weights
    total_weight = np.sum(avg_coeffs)
    normalized_weights = (avg_coeffs / total_weight) * 100
    
    weights_df = pd.DataFrame({
        'Factor': ['Rating', 'H2H', 'Style'],
        'Learned Weight (%)': normalized_weights
    })
    
    return weights_df.sort_values(by='Learned Weight (%)', ascending=False)


# --- Main Execution Block ---
if __name__ == "__main__":
    
    try:
        player_features_df = pd.read_csv('player_features.csv', index_col='player_name')
        all_games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: 'player_features.csv' or 'all_games.csv' not found.")
        print("Please run the 'engineering.py' script first.")
        exit()
        
    # Find the robust weights using the new cross-validation method
    robust_weights_df = find_robust_weights(all_games_df, player_features_df)
    
    print("--- Step 3 of 3: Presenting robust results... ---")
    print("\n\n===================================================================")
    print("            ROBUST DATA-DRIVEN SIMULATION WEIGHTS")
    print("===================================================================")
    print("The ML model used Cross-Validation to prevent data leakage and find")
    print("the most reliable predictive factors for determining a game's winner.\n")
    
    print(robust_weights_df.round(1).to_string(index=False))
    
    print("\n--- Recommendation ---")
    print("These weights are much more trustworthy and realistic.")
    print("Update the weights in 'simulation_advanced.py' with these values")
    print("to run the most accurate, data-driven tournament simulation.")
    print("===================================================================")