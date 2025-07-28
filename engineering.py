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

# ==============================================================================
# DATA PROCESSING FUNCTIONS (UNCHANGED)
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
                        if 'players' in tournament:
                            for player in tournament['players']:
                                player_name = player.get('name')
                                if player_name:
                                    all_arena_stats.append({
                                        'player_name': player_name.lower(), 'score': player.get('score'),
                                        'games_played': len(player.get('sheet', {}).get('scores', '')),
                                        'sheet': player.get('sheet', {}).get('scores', ''), 'source_file': filename
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
    for game in all_games:
        if game.get('winner_name'): game['winner_name'] = game['winner_name'].lower()
        if game.get('loser_name'): game['loser_name'] = game['loser_name'].lower()
    games_df = pd.DataFrame(all_games)
    if not games_df.empty:
        games_df = games_df[games_df['status'] != 'Abandoned'].copy()
    
    return games_df, arena_df


def calculate_performance_z_scores(arena_df):
    """Calculates a normalized Performance Z-Score for each player in each tournament."""
    print("--- Step 2 of 4: Calculating Performance Z-Scores ---")
    
    z_scores = []
    for file, tournament_group in arena_df.groupby('source_file'):
        valid_players = tournament_group[tournament_group['games_played'] >= MIN_GAMES_PER_TOURNAMENT].copy()
        if len(valid_players) < 2: continue
        valid_players['ppg'] = np.divide(valid_players['score'], valid_players['games_played'])
        mean_ppg, std_dev_ppg = valid_players['ppg'].mean(), valid_players['ppg'].std()
        if std_dev_ppg == 0: valid_players['z_score'] = 0
        else: valid_players['z_score'] = (valid_players['ppg'] - mean_ppg) / std_dev_ppg
        z_scores.extend(valid_players[['player_name', 'z_score']].to_dict('records'))
    return pd.DataFrame(z_scores)

# ==============================================================================
# UPDATED FUNCTION WITH THE FIX
# ==============================================================================
def create_player_profiles(games_df, z_scores_df, arena_df, qualified_players):
    """Calculates all statistics and returns the final report DataFrame and H2H dictionary."""
    print("--- Step 3 of 4: Calculating Player Statistics and H2H Rivalries ---")
    
    player_profiles = []
    for player_name in qualified_players:
        profile = {'player_name': player_name}
        player_games_as_winner = games_df[games_df['winner_name'] == player_name]
        player_all_games = pd.concat([player_games_as_winner, games_df[games_df['loser_name'] == player_name]])
        player_arena_data = arena_df[arena_df['player_name'] == player_name]
        player_z_scores = z_scores_df[z_scores_df['player_name'] == player_name]

        if not player_z_scores.empty and len(player_z_scores) > 1: profile['z_score_volatility'] = player_z_scores['z_score'].std()
        else: profile['z_score_volatility'] = None
            
        total_games = len(player_all_games)
        win_rate_on_time = ((player_games_as_winner['status'] == 'Time forfeit').sum() / total_games) * 100 if total_games > 0 else 0
        loss_rate_on_time = ((player_all_games['loser_name'] == player_name) & (player_all_games['status'] == 'Time forfeit')).sum() / total_games * 100 if total_games > 0 else 0
        profile['time_pressure_factor'] = win_rate_on_time - loss_rate_on_time
        
        if not player_arena_data.empty:
            total_games_in_arena = player_arena_data['games_played'].sum()
            berserk_wins = player_arena_data['sheet'].str.count('3').sum() + player_arena_data['sheet'].str.count('5').sum()
            profile['berserk_win_rate'] = (berserk_wins / total_games_in_arena) * 100 if total_games_in_arena > 0 else 0
        else: profile['berserk_win_rate'] = 0
            
        valid_arena_data = player_arena_data[player_arena_data['games_played'] >= MIN_GAMES_PER_TOURNAMENT]
        profile['avg_points_per_game'] = np.divide(valid_arena_data['score'], valid_arena_data['games_played']).mean() if not valid_arena_data.empty else 0
        profile['avg_games_per_tournament'] = valid_arena_data['games_played'].mean() if not valid_arena_data.empty else 0
            
        giant_slayings = (player_games_as_winner['loser_rating'] - player_games_as_winner['winner_rating'] >= GIANT_SLAYER_RATING_DIFF).sum()
        total_wins = len(player_games_as_winner)
        profile['giant_slayer_rate'] = (giant_slayings / total_wins) * 100 if total_wins > 0 else 0
        player_profiles.append(profile)

    profiles_df = pd.DataFrame(player_profiles).set_index('player_name').fillna(0)
    
    h2h_groups = games_df.groupby(['winner_name', 'loser_name']).size().reset_index(name='wins')
    h2h_data = {}
    for player_name in qualified_players:
        h2h_data[player_name] = {}
        wins_as_p1 = h2h_groups[h2h_groups['winner_name'] == player_name]
        losses_as_p1 = h2h_groups[h2h_groups['loser_name'] == player_name]
        opponents = pd.concat([wins_as_p1['loser_name'], losses_as_p1['winner_name']]).unique()
        for opponent in opponents:
            if opponent in qualified_players:
                wins = wins_as_p1[wins_as_p1['loser_name'] == opponent]['wins'].sum()
                losses = losses_as_p1[losses_as_p1['winner_name'] == opponent]['wins'].sum()
                h2h_data[player_name][opponent] = {'wins': wins, 'losses': losses}

    favorite_opponents, nemesis_players = {}, {}
    for player, opponents in h2h_data.items():
        rivalry_stats = []
        for opponent, record in opponents.items():
            total_games = record['wins'] + record['losses']
            if total_games >= H2H_MIN_GAMES_FOR_RIVALRY:
                win_pct = (record['wins'] / total_games) * 100
                rivalry_stats.append({'opponent': opponent, 'win_pct': win_pct})
        if rivalry_stats:
            sorted_rivals = sorted(rivalry_stats, key=lambda x: x['win_pct'], reverse=True)
            favorite_opponents[player] = [f"{r['opponent']} ({r['win_pct']:.0f}%)" for r in sorted_rivals[:3]]
            nemesis_players[player] = [f"{r['opponent']} ({r['win_pct']:.0f}%)" for r in sorted(rivalry_stats, key=lambda x: x['win_pct'])[:3]]
    
    # --- THE FIX IS HERE ---
    # Step 1: Map the dictionaries to the DataFrame. This will create NaN for players with no rivals.
    profiles_df['Favorite Opponents'] = profiles_df.index.map(favorite_opponents)
    profiles_df['Nemesis Players'] = profiles_df.index.map(nemesis_players)
    
    # Step 2: Use .apply() on the new COLUMNS to replace NaN with an empty list. This is the correct method.
    profiles_df['Favorite Opponents'] = profiles_df['Favorite Opponents'].apply(lambda x: x if isinstance(x, list) else [])
    profiles_df['Nemesis Players'] = profiles_df['Nemesis Players'].apply(lambda x: x if isinstance(x, list) else [])
    # --- END OF FIX ---

    for col in ['berserk_win_rate', 'z_score_volatility', 'time_pressure_factor']:
        min_val, max_val = profiles_df[col].min(), profiles_df[col].max()
        if (max_val - min_val) > 0: profiles_df[f'norm_{col}'] = (profiles_df[col] - min_val) / (max_val - min_val)
        else: profiles_df[f'norm_{col}'] = 0.5
            
    report_df = pd.DataFrame(index=profiles_df.index)
    report_df['Consistency'] = 1 - profiles_df['norm_z_score_volatility']
    report_df['Speed'] = profiles_df['norm_time_pressure_factor']
    report_df['Aggressiveness'] = profiles_df['norm_berserk_win_rate']
    report_df['Avg Points Per Game'] = profiles_df['avg_points_per_game']
    report_df['Avg Games (Stamina)'] = profiles_df['avg_games_per_tournament']
    report_df['Giant-Slayer Rate'] = profiles_df['giant_slayer_rate']
    report_df['Favorite Opponents'] = profiles_df['Favorite Opponents']
    report_df['Nemesis Players'] = profiles_df['Nemesis Players']

    return report_df, h2h_data

# ==============================================================================
# INTERACTIVE FUNCTIONS (UNCHANGED)
# ==============================================================================
def display_single_player_report(player_name, report_df):
    """Formats and prints the full scouting report for one player."""
    if player_name not in report_df.index:
        print(f"\n>> ERROR: Player '{player_name}' not found in the qualified dataset.")
        return
    player_data = report_df.loc[player_name]
    print("\n\n==========================================================")
    print(f"            SCOUTING REPORT: {player_name.upper()}")
    print("==========================================================")
    print("\n--- Core Archetype (Normalized 0.0 to 1.0) ---")
    print(f"  Consistency (Reliability):       {player_data['Consistency']:.2f}")
    print(f"  Speed (Clock Management):        {player_data['Speed']:.2f}")
    print(f"  Aggressiveness (Berserk Style):  {player_data['Aggressiveness']:.2f}")
    print("\n--- Performance Averages ---")
    print(f"  Average Points Per Game:         {player_data['Avg Points Per Game']:.2f}")
    print(f"  Average Games Per Tournament:    {player_data['Avg Games (Stamina)']:.1f}")
    print(f"  Giant-Slayer Rate (%):           {player_data['Giant-Slayer Rate']:.1f}%")
    print("\n--- Head-to-Head Intel ---")
    fav_opps, nem_opps = player_data.get('Favorite Opponents', []), player_data.get('Nemesis Players', [])
    print(f"  Favorite Opponents (Top 3):      {', '.join(fav_opps) if fav_opps else 'N/A'}")
    print(f"  Nemesis Players (Top 3):         {', '.join(nem_opps) if nem_opps else 'N/A'}")
    print("----------------------------------------------------------\n")

def display_h2h_comparison(p1, p2, report_df, h2h_data):
    """Formats and prints a side-by-side comparison of two players."""
    if p1 not in report_df.index or p2 not in report_df.index:
        print(f"\n>> ERROR: One or both players not found. Please check names.")
        return
    p1_wins, p2_wins = h2h_data.get(p1, {}).get(p2, {}).get('wins', 0), h2h_data.get(p2, {}).get(p1, {}).get('wins', 0)
    print("\n\n==========================================================")
    print(f"        HEAD-TO-HEAD: {p1.upper()} vs {p2.upper()}")
    print("==========================================================")
    print(f"\n  Direct Rivalry Score: {p1} {p1_wins} - {p2_wins} {p2}\n")
    comparison_df = report_df.loc[[p1, p2]].T
    comparison_df.columns = [name.upper() for name in comparison_df.columns]
    for col in ['Consistency', 'Speed', 'Aggressiveness', 'Avg Points Per Game', 'Avg Games (Stamina)']:
        if col in comparison_df.index: comparison_df.loc[col] = comparison_df.loc[col].apply(lambda x: f"{x:.2f}")
    if 'Giant-Slayer Rate' in comparison_df.index:
        comparison_df.loc['Giant-Slayer Rate'] = comparison_df.loc['Giant-Slayer Rate'].apply(lambda x: f"{x:.1f}%")
    print(comparison_df.drop(['Favorite Opponents', 'Nemesis Players']))
    print("----------------------------------------------------------\n")

# ==============================================================================
# MAIN EXECUTION BLOCK (UNCHANGED)
# ==============================================================================
if __name__ == "__main__":
    
    games_df, arena_df = load_and_process_data()
    if not games_df.empty:
        z_scores_df = calculate_performance_z_scores(arena_df)
        player_game_counts = games_df['winner_name'].value_counts().add(games_df['loser_name'].value_counts(), fill_value=0)
        qualified_players = player_game_counts[player_game_counts >= MIN_GAMES_THRESHOLD_OVERALL].index.tolist()
        print("--- Step 4 of 4: Building Final Report Database ---")
        report_df, h2h_data = create_player_profiles(games_df, z_scores_df, arena_df, qualified_players)
        print("\nDatabase ready. Launching interactive terminal.")
        
        while True:
            print("\n==============================================")
            print("    INTERACTIVE PLAYER SCOUTING TERMINAL")
            print("==============================================")
            print("  1: Look up a single player")
            print("  2: Compare two players (H2H)")
            print("  Type 'exit' or 'quit' to end.")
            choice = input(">> Enter your choice: ").strip()
            if choice == '1':
                player_name = input("   Enter player name: ").lower().strip()
                display_single_player_report(player_name, report_df)
            elif choice == '2':
                names_input = input("   Enter two names separated by a space: ").lower().strip()
                names = names_input.split()
                if len(names) == 2: display_h2h_comparison(names[0], names[1], report_df, h2h_data)
                else: print("\n>> ERROR: Please enter exactly two names.")
            elif choice.lower() in ['exit', 'quit', '3']:
                print("\nExiting scouting terminal. Goodbye!")
                break
            else:
                print("\n>> ERROR: Invalid choice. Please enter 1, 2, or 'exit'.")
    else:
        print("\nNo individual game data found. Cannot create features.")