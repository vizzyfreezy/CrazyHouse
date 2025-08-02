import os
import json
import pandas as pd
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
ARENA_DATA_FOLDER = './tournament_cache'
GAMES_DATA_FOLDER = './detailed_games'
MIN_GAMES_THRESHOLD_OVERALL = 50
MIN_GAMES_PER_TOURNAMENT = 5
H2H_MIN_GAMES_FOR_RIVALRY = 10
GIANT_SLAYER_RATING_DIFF = 200

# --- USERNAME MAPPING (Case-Insensitive) ---
# Add alternative usernames here to map them to a primary account.
# All usernames will be converted to and stored in lowercase.
# Format: "alternative_username": "primary_username"
USERNAME_ALIAS_TO_PRIMARY = {
    "dabee": "warlock_dabee",
    "nevagivup": "clintonsalako",
    "neo-matrix": "hardeywale",
    "noblechuks_cno": "patzernoblechuks",
    "specialagentfash": "ezengwori",
    "naijacobweb":"crazybugg",
}

# Create a mapping where keys and values are lowercase
USERNAME_MAPPING = {k.lower(): v.lower() for k, v in USERNAME_ALIAS_TO_PRIMARY.items()}
# Also map the primary username to itself (lowercase) to handle cases where it appears directly.
for primary_name in USERNAME_ALIAS_TO_PRIMARY.values():
    USERNAME_MAPPING[primary_name.lower()] = primary_name.lower()

# ==============================================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================================

def load_and_process_data():
    """Loads and processes all data from both folders."""
    print("--- Step 1 of 4: Loading and Processing Raw Data Files ---")
    
    all_arena_stats = []
    if os.path.exists(ARENA_DATA_FOLDER):
        for filename in os.listdir(ARENA_DATA_FOLDER):
            if filename.endswith('.json'):
                file_path = os.path.join(ARENA_DATA_FOLDER, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        tournament = json.load(f)
                        player_data_for_this_file = {}
                        if 'players' in tournament:
                            for player in tournament['players']:
                                player_name = player.get('name')
                                if player_name:
                                    player_name_lower = player_name.lower()
                                    player_data_for_this_file[player_name_lower] = {
                                        'score': player.get('score'), 'games_played': len(player.get('sheet', {}).get('scores', '')),
                                        'sheet': player.get('sheet', {}).get('scores', ''), 'rating_at_event': player.get('rating'),
                                        'performance_rating': None
                                    }
                        if 'podium' in tournament:
                            for player in tournament['podium']:
                                player_name = player.get('name')
                                if player_name:
                                    player_name_lower = player_name.lower()
                                    if player_name_lower in player_data_for_this_file:
                                        player_data_for_this_file[player_name_lower]['performance_rating'] = player.get('performance')
                        for name, data in player_data_for_this_file.items():
                            true_perf = data['performance_rating']
                            if true_perf is None: true_perf = data['rating_at_event']
                            all_arena_stats.append({
                                'player_name': name, 'score': data['score'], 'games_played': data['games_played'], 
                                'sheet': data['sheet'], 'true_performance': true_perf, 'source_file': filename
                            })
                    except json.JSONDecodeError: print(f"Warning: Could not decode JSON from {filename}")
    arena_df = pd.DataFrame(all_arena_stats)

    all_games = []
    if os.path.exists(GAMES_DATA_FOLDER):
        for filename in os.listdir(GAMES_DATA_FOLDER):
             if filename.endswith('.json'):
                file_path = os.path.join(GAMES_DATA_FOLDER, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        games_list = json.load(f)
                        if isinstance(games_list, list): all_games.extend(games_list)
                    except json.JSONDecodeError: print(f"Warning: Could not decode JSON from {filename}")
    
    # --- THE FIX IS HERE ---
    # Loop through games to standardize names AND create the 'datetime' column
    for game in all_games:
        if game.get('winner_name'): game['winner_name'] = game['winner_name'].lower()
        if game.get('loser_name'): game['loser_name'] = game['loser_name'].lower()
        # This line was missing and is now restored. It combines 'date' and 'time'.
        if 'date' in game and 'time' in game:
            game['datetime'] = datetime.strptime(f"{game['date']} {game['time']}", '%Y.%m.%d %H:%M:%S')
    # --- END OF FIX ---
            
    games_df = pd.DataFrame(all_games)
    if not games_df.empty:
        games_df = games_df[games_df['status'] != 'Abandoned'].copy()
    
    # --- APPLY USERNAME MAPPING ---
    # Consolidate usernames based on the mapping.
    # Usernames are already lowercased during the initial processing.
    if not arena_df.empty:
        arena_df['player_name'] = arena_df['player_name'].map(USERNAME_MAPPING).fillna(arena_df['player_name'])
    if not games_df.empty:
        games_df['winner_name'] = games_df['winner_name'].map(USERNAME_MAPPING).fillna(games_df['winner_name'])
        games_df['loser_name'] = games_df['loser_name'].map(USERNAME_MAPPING).fillna(games_df['loser_name'])
        
    return games_df, arena_df


def calculate_performance_metrics(arena_df):
    """
    Calculates normalized performance scores (Z-Score and Normalized PPG) for each player.
    Normalization is done *within* each tournament to ensure fairness.
    """
    print("--- Step 2 of 4: Calculating Performance Metrics ---")
    
    all_z_scores = []
    all_normalized_ppg = []

    for file, tournament_group in arena_df.groupby('source_file'):
        # Filter for players who have played a minimum number of games in the tournament
        valid_players = tournament_group[tournament_group['games_played'] >= MIN_GAMES_PER_TOURNAMENT].copy()
        if len(valid_players) < 2:
            continue  # Need at least two players to calculate meaningful stats

        # Calculate Points Per Game (PPG)
        valid_players['ppg'] = np.divide(valid_players['score'], valid_players['games_played'])
        
        # --- Z-Score Calculation (for Consistency) ---
        mean_ppg, std_dev_ppg = valid_players['ppg'].mean(), valid_players['ppg'].std()
        if std_dev_ppg == 0:
            valid_players['z_score'] = 0
        else:
            valid_players['z_score'] = (valid_players['ppg'] - mean_ppg) / std_dev_ppg
        all_z_scores.append(valid_players[['player_name', 'z_score']])

        # --- Normalized PPG Calculation (for True Performance) ---
        min_ppg, max_ppg = valid_players['ppg'].min(), valid_players['ppg'].max()
        if (max_ppg - min_ppg) > 0:
            valid_players['norm_ppg'] = (valid_players['ppg'] - min_ppg) / (max_ppg - min_ppg)
        else:
            valid_players['norm_ppg'] = 0.5  # Assign a neutral score if all players had the same PPG
        all_normalized_ppg.append(valid_players[['player_name', 'norm_ppg']])

    z_scores_df = pd.concat(all_z_scores, ignore_index=True) if all_z_scores else pd.DataFrame(columns=['player_name', 'z_score'])
    norm_ppg_df = pd.concat(all_normalized_ppg, ignore_index=True) if all_normalized_ppg else pd.DataFrame(columns=['player_name', 'norm_ppg'])
    
    return z_scores_df, norm_ppg_df


def create_player_profiles(games_df, z_scores_df, norm_ppg_df, arena_df, qualified_players):
    """Calculates all statistics and returns the final report DataFrame."""
    print("--- Step 3 of 4: Calculating Player Statistics and H2H Rivalries ---")
    
    # --- Aggregate the normalized scores for each player ---
    player_avg_norm_ppg = norm_ppg_df.groupby('player_name')['norm_ppg'].mean().reset_index()
    player_avg_norm_ppg = player_avg_norm_ppg.set_index('player_name')

    player_profiles = []
    for player_name in qualified_players:
        profile = {'player_name': player_name}
        player_games_as_winner = games_df[games_df['winner_name'] == player_name]
        player_all_games = pd.concat([player_games_as_winner, games_df[games_df['loser_name'] == player_name]])
        player_arena_data = arena_df[arena_df['player_name'] == player_name]
        player_z_scores = z_scores_df[z_scores_df['player_name'] == player_name]

        # This line will now work correctly because 'datetime' exists
        last_game = player_all_games.sort_values('datetime').iloc[-1]
        profile['current_rating'] = last_game['winner_rating'] if last_game['winner_name'] == player_name else last_game['loser_rating']

        if not player_z_scores.empty and len(player_z_scores) > 1:
            profile['z_score_volatility'] = player_z_scores['z_score'].std()
        else:
            profile['z_score_volatility'] = None
            
        total_games = len(player_all_games)
        win_rate_on_time = ((player_games_as_winner['status'] == 'Time forfeit').sum() / total_games) * 100 if total_games > 0 else 0
        loss_rate_on_time = ((player_all_games['loser_name'] == player_name) & (player_all_games['status'] == 'Time forfeit')).sum() / total_games * 100 if total_games > 0 else 0
        profile['time_pressure_factor'] = win_rate_on_time - loss_rate_on_time
        
        if not player_arena_data.empty:
            total_games_in_arena = player_arena_data['games_played'].sum()
            berserk_wins = player_arena_data['sheet'].str.count('3').sum() + player_arena_data['sheet'].str.count('5').sum()
            profile['berserk_win_rate'] = (berserk_wins / total_games_in_arena) * 100 if total_games_in_arena > 0 else 0
            # The new, fairer performance metric
            profile['true_avg_performance'] = player_avg_norm_ppg.loc[player_name, 'norm_ppg'] if player_name in player_avg_norm_ppg.index else 0
        else:
            profile['berserk_win_rate'], profile['true_avg_performance'] = 0, 0
            
        valid_arena_data = player_arena_data[player_arena_data['games_played'] >= MIN_GAMES_PER_TOURNAMENT]
        profile['avg_points_per_game'] = np.divide(valid_arena_data['score'], valid_arena_data['games_played']).mean() if not valid_arena_data.empty else 0
        profile['avg_games_per_tournament'] = valid_arena_data['games_played'].mean() if not valid_arena_data.empty else 0
            
        giant_slayings = (player_games_as_winner['loser_rating'] - player_games_as_winner['winner_rating'] >= GIANT_SLAYER_RATING_DIFF).sum()
        total_wins = len(player_games_as_winner)
        profile['giant_slayer_rate'] = (giant_slayings / total_wins) * 100 if total_wins > 0 else 0
        player_profiles.append(profile)

    profiles_df = pd.DataFrame(player_profiles).set_index('player_name').fillna(0)
    
    h2h_groups = games_df.groupby(['winner_name', 'loser_name']).size().reset_index(name='wins')
    favorite_opponents, nemesis_players = {}, {}
    for player_name in qualified_players:
        wins_as_p1 = h2h_groups[h2h_groups['winner_name'] == player_name]
        losses_as_p1 = h2h_groups[h2h_groups['loser_name'] == player_name]
        rivalry_stats = []
        opponents = pd.concat([wins_as_p1['loser_name'], losses_as_p1['winner_name']]).unique()
        for opponent in opponents:
            if opponent in qualified_players:
                wins, losses = wins_as_p1[wins_as_p1['loser_name'] == opponent]['wins'].sum(), losses_as_p1[losses_as_p1['winner_name'] == opponent]['wins'].sum()
                if (wins + losses) >= H2H_MIN_GAMES_FOR_RIVALRY:
                    rivalry_stats.append({'opponent': opponent, 'win_pct': (wins / (wins + losses)) * 100})
        if rivalry_stats:
            sorted_rivals = sorted(rivalry_stats, key=lambda x: x['win_pct'], reverse=True)
            favorite_opponents[player_name] = [f"{r['opponent']} ({r['win_pct']:.0f}%)" for r in sorted_rivals[:3]]
            nemesis_players[player_name] = [f"{r['opponent']} ({r['win_pct']:.0f}%)" for r in sorted(rivalry_stats, key=lambda x: x['win_pct'])[:3]]
    
    profiles_df['Favorite Opponents'] = profiles_df.index.map(favorite_opponents)
    profiles_df['Nemesis Players'] = profiles_df.index.map(nemesis_players)
    profiles_df['Favorite Opponents'] = profiles_df['Favorite Opponents'].apply(lambda x: x if isinstance(x, list) else [])
    profiles_df['Nemesis Players'] = profiles_df['Nemesis Players'].apply(lambda x: x if isinstance(x, list) else [])

    for col in ['berserk_win_rate', 'z_score_volatility', 'time_pressure_factor']:
        min_val, max_val = profiles_df[col].min(), profiles_df[col].max()
        if (max_val - min_val) > 0:
            profiles_df[f'norm_{col}'] = (profiles_df[col] - min_val) / (max_val - min_val)
        else:
            profiles_df[f'norm_{col}'] = 0.5
            
    report_df = pd.DataFrame(index=profiles_df.index)
    report_df['Current Rating'] = profiles_df['current_rating']
    report_df['Consistency'] = 1 - profiles_df['norm_z_score_volatility']
    report_df['Speed'] = profiles_df['norm_time_pressure_factor']
    report_df['Aggressiveness'] = profiles_df['norm_berserk_win_rate']
    report_df['Avg Points Per Game'] = profiles_df['avg_points_per_game']
    report_df['Avg Games (Stamina)'] = profiles_df['avg_games_per_tournament']
    report_df['Giant-Slayer Rate'] = profiles_df['giant_slayer_rate']
    report_df['Berserk Win Rate'] = profiles_df['berserk_win_rate']
    report_df['Favorite Opponents'] = profiles_df['Favorite Opponents']
    report_df['Nemesis Players'] = profiles_df['Nemesis Players']
    report_df['True Avg Performance'] = profiles_df['true_avg_performance']
    report_df['Consistency_norm'] = 1 - profiles_df['norm_z_score_volatility']
    report_df['Speed_norm'] = profiles_df['norm_time_pressure_factor']
    report_df['Aggressiveness_norm'] = profiles_df['norm_berserk_win_rate']


    return report_df


def rank_players_by_strategy(report_df, weights):
    """Calculates a 'Suitability Score' for each player based on a given weighting strategy."""
    norm_df = pd.DataFrame(index=report_df.index)
    norm_df['Consistency'], norm_df['Speed'], norm_df['Aggressiveness'] = report_df['Consistency'], report_df['Speed'], report_df['Aggressiveness']
    
    for col in ['Avg Points Per Game', 'Avg Games (Stamina)', 'Giant-Slayer Rate', 'True Avg Performance']:
        min_val, max_val = report_df[col].min(), report_df[col].max()
        if (max_val - min_val) > 0: norm_df[col] = (report_df[col] - min_val) / (max_val - min_val)
        else: norm_df[col] = 0.5
            
    suitability_score = pd.Series(0.0, index=norm_df.index)
    for characteristic, weight in weights.items():
        if characteristic in norm_df.columns: suitability_score += norm_df[characteristic] * weight
            
    ranked_df = report_df.copy()
    ranked_df['Suitability Score'] = suitability_score
    return ranked_df.sort_values(by='Suitability Score', ascending=False)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    games_df, arena_df = load_and_process_data()

    if not games_df.empty:
        z_scores_df, norm_ppg_df = calculate_performance_metrics(arena_df)
        
        player_game_counts = games_df['winner_name'].value_counts().add(games_df['loser_name'].value_counts(), fill_value=0)
        qualified_players = player_game_counts[player_game_counts >= MIN_GAMES_THRESHOLD_OVERALL].index.tolist()
        
        print("--- Step 4 of 4: Generating Final Reports ---")
        
        report_df = create_player_profiles(games_df, z_scores_df, norm_ppg_df, arena_df, qualified_players)
        
        # Output 1: The Main Player Scouting Report
        print("\n\n=======================================================================================================")
        print("                                     PLAYER SCOUTING REPORT")
        print("=======================================================================================================")
        pd.set_option('display.max_rows', 200)
        pd.set_option('display.width', 180)
        pd.set_option('display.max_colwidth', 40)
        
        final_columns = [
            'Current Rating', 'True Avg Performance', 'Consistency', 'Speed', 'Aggressiveness', 
            'Avg Points Per Game', 'Avg Games (Stamina)', 'Giant-Slayer Rate', 
            'Favorite Opponents', 'Nemesis Players','Berserk Win Rate'
        ]
        final_columns = [col for col in final_columns if col in report_df.columns]
        print(report_df[final_columns].sort_values(by='True Avg Performance', ascending=False).round(2))
        
        # Output 2: The Strategic Ranking Report
        print("\n\n=======================================================================================================")
        print("                                     STRATEGIC TEAM RANKING")
        print("=======================================================================================================")
        
        strategy_weights = {
            'True Avg Performance':      0.4, 'Consistency':               0.2, 
            'Avg Games (Stamina)':       0.2, 'Speed':                     0.2, 
            'Aggressiveness':            0.0, 'Giant-Slayer Rate':         0.0,
        }
        
        print(f"Ranking players based on the defined strategy weights...")
        ranked_report = rank_players_by_strategy(report_df, strategy_weights)
        
        ranked_display_cols = [
            'Suitability Score', 'True Avg Performance', 'Consistency', 'Speed', 'Aggressiveness', 
            'Avg Points Per Game', 'Avg Games (Stamina)', 'Giant-Slayer Rate','Berserk Win Rate'
        ]
        ranked_display_cols = [col for col in ranked_display_cols if col in ranked_report.columns]
        print(ranked_report[ranked_display_cols].head(15).round(2))
        print("=======================================================================================================")
        
        print("\nSaving final player data to 'player_features.csv' for simulation...")
        report_df.to_csv('player_features.csv')
        games_df.to_csv('all_games.csv', index=False)
        print("Save complete.")

    else:
        print("\nNo individual game data found. Cannot create features.")