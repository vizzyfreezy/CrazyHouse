import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
from multiprocessing import Pool
import os

# Import all necessary simulation logic and constants from simulate_teambattle.py
from simulate_teambattle import (
    calculate_h2h_advantage,
    calculate_style_advantage,
    simulate_game_advanced,
    build_h2h_wins_dict_from_games, # Corrected import
    WEIGHT_RATING, WEIGHT_H2H, WEIGHT_STYLE, DRAW_PROBABILITY,
    INFAMY_RATING_THRESHOLD, DEFAULT_CLIP_RANGE,
    INFAMOUS_CLIP_RANGE_TIGHT, INFAMOUS_CLIP_RANGE_VERY_TIGHT,
    RATING_GAP_THRESHOLD_1, RATING_GAP_THRESHOLD_2
)

# ==============================================================================
# DRAFT VALIDATOR CONFIGURATION - EDIT THIS SECTION
# ==============================================================================

# List all players available at the START of the draft.
PLAYER_POOL = [
    'anthonyoja', 'zgm-giantkiller', 'bayormiller_cno', 'trappatoni', 'zlater007', 'b4elma',
    'patzernoblechuks', 'ovokodigood', 'adet2510', 'noblechuks_cno', 'spicypearl8', 'tommybrooks',
    'ezengwori', 'ageless_2', 'crazybugg', 'warlock_dabee', 'hardeywale', 'vegakov',
    'kirekachesschamp', 'bb_thegame', 'martins177', 'lexzero2', 'overgiftedlight', 'nevagivup'
]

# --- Simulation Parameters for the Final Showdown ---
# High number for a confident validation result.
FINAL_SHOWDOWN_SIMULATIONS = 10000

# Parameters for the draft analysis itself
SIMULATIONS_PER_CANDIDATE = 500
GAMES_PER_MATCHUP = 2

# ==============================================================================
# CORE SIMULATION & DRAFT LOGIC (DO NOT EDIT)
# ==============================================================================

def run_mini_simulation(my_team, opponent_team, player_data_dict, h2h_wins_dict):
    if not opponent_team: return {p: 0 for p in my_team}
    all_player_results = []
    for _ in range(SIMULATIONS_PER_CANDIDATE):
        player_scores = {player: 0.0 for player in my_team + opponent_team}
        for p1_name, p2_name in itertools.product(my_team, opponent_team):
            for _ in range(GAMES_PER_MATCHUP):
                p1_stats = player_data_dict[p1_name]
                p2_stats = player_data_dict[p2_name]
                p1_score, p2_score = simulate_game_advanced(p1_name, p2_name, p1_stats, p2_stats, h2h_wins_dict)
                player_scores[p1_name] += p1_score
                player_scores[p2_name] += p2_score
        all_player_results.append(pd.Series(player_scores))
    return pd.DataFrame(all_player_results).mean(axis=0)

def analyze_candidate_wrapper(args):
    candidate, my_team, opponent_team, player_data_dict, h2h_wins_dict = args
    test_my_team = my_team + [candidate]
    avg_scores = run_mini_simulation(test_my_team, opponent_team, player_data_dict, h2h_wins_dict)
    return candidate, avg_scores.get(candidate, 0)

def find_best_pick_parallel(my_team, opponent_team, player_pool, player_data_dict, h2h_wins_dict):
    tasks = [(candidate, my_team, opponent_team, player_data_dict, h2h_wins_dict) for candidate in player_pool]
    num_processes = os.cpu_count() if os.cpu_count() else 4 # Fallback to 4 if not detectable
    with Pool(processes=num_processes) as p:
        results = list(tqdm(p.imap_unordered(analyze_candidate_wrapper, tasks), total=len(tasks), desc="Analyzing Pick", leave=False))
    return sorted(results, key=lambda item: item[1], reverse=True)[0][0]

def build_h2h_data_from_games(games_df):
    return build_h2h_wins_dict_from_games(games_df)

def run_final_showdown_simulation(teams, player_data_dict, h2h_wins_dict):
    team_scores = {team_name: 0.0 for team_name in teams}
    for p1_name, p2_name in itertools.product(teams['Smart Team'], teams['Random Team']):
        for _ in range(GAMES_PER_MATCHUP):
            p1_stats = player_data_dict[p1_name]
            p2_stats = player_data_dict[p2_name]
            p1_score, p2_score = simulate_game_advanced(p1_name, p2_name, p1_stats, p2_stats, h2h_wins_dict)
            team_scores['Smart Team'] += p1_score
            team_scores['Random Team'] += p2_score
    return pd.Series(team_scores)

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv')
        print("--- Player and game data loaded successfully. Database is ready. ---")
    except FileNotFoundError:
        print("\n!!! ERROR: 'player_features.csv' or 'all_games.csv' not found. Please run 'engineering.py' first.")
        exit()

    player_data_dict = player_data_df.to_dict('index') # Convert to dict once
    h2h_wins_dict = build_h2h_data_from_games(games_df)

    player_pool = sorted([p.lower() for p in PLAYER_POOL])
    smart_team, random_team = [], []
    
    missing_players = [p for p in player_pool if p not in player_data_dict]
    if missing_players:
        print(f"\n!!! ERROR: The following players in your POOL are not in the data file: {missing_players}")
        exit()
        
    
    # --- 1. Automated Draft ---
    print("\n--- Starting Automated Draft: Smart vs. Random ---")
    turn = 0
    draft_progress = tqdm(total=len(player_pool), desc="Drafting Players")
    while player_pool:
        if turn % 2 == 0: # Smart Team's turn
            best_pick = find_best_pick_parallel(smart_team, random_team, player_pool, player_data_dict, h2h_wins_dict)
            smart_team.append(best_pick)
            player_pool.remove(best_pick)
        else: # Random Team's turn
            # To make it slightly more realistic, let's assume random picks the best remaining by rating
            # This is a tougher test than pure random.
            
            # Filter player_data_df for remaining players and find highest rated
            remaining_players_df = player_data_df.loc[player_pool]
            random_pick = remaining_players_df['Current Rating'].idxmax() # Picks highest rated player left
            
            random_team.append(random_pick)
            player_pool.remove(random_pick)
        turn += 1
        draft_progress.update(1)
    draft_progress.close()
    
    print("\n--- Draft Complete ---")
    print(f"  Smart Team: {', '.join(smart_team)}")
    print(f"  Random Team (Best Available Rating): {', '.join(random_team)}")
    
    # --- 2. Post-Draft Showdown ---
    print(f"\n--- Running Final Showdown ({FINAL_SHOWDOWN_SIMULATIONS} simulations) ---")
    final_teams = {"Smart Team": smart_team, "Random Team": random_team}
    
    all_results = []
    for _ in tqdm(range(FINAL_SHOWDOWN_SIMULATIONS), desc="Running Showdown"):
        final_scores = run_final_showdown_simulation(final_teams, player_data_dict, h2h_wins_dict)
        all_results.append(final_scores)
    
    results_df = pd.DataFrame(all_results)
    
    # --- 3. The Verdict ---
    winners = []
    for i in range(len(results_df)):
        row = results_df.iloc[i]
        if (row == row.max()).sum() > 1: winners.append('Draw')
        else: winners.append(row.idxmax())
    win_percentages = (pd.Series(winners).value_counts() / FINAL_SHOWDOWN_SIMULATIONS * 100).fillna(0)

    print("\n\n=======================================================================")
    print("                DRAFT ASSISTANT VALIDATION REPORT")
    print("=======================================================================")
    print(f"Based on a draft between a 'Smart' team (using the assistant) and a")
    print(f"'Random' team (picking the best available rating), followed by")
    print(f"{FINAL_SHOWDOWN_SIMULATIONS} simulated head-to-head battles.\n")
    
    report_df = pd.DataFrame({'Win %': win_percentages})
    report_df['Avg Team Score'] = results_df.mean(axis=0)
    
    print("--- Validation Results ---")
    print(report_df.sort_values(by='Win %', ascending=False).round(1).to_string())
    
    print("\n--- Verdict ---")
    smart_win_pct = report_df.loc['Smart Team', 'Win %']
    if smart_win_pct > 55:
        print("SUCCESS: The Smart Draft Assistant provides a significant competitive advantage.")
    elif smart_win_pct > 50:
        print("VALIDATED: The Smart Draft Assistant provides a slight edge.")
    else:
        print("INCONCLUSIVE: The Smart Draft Assistant's picks did not prove to be better than simply picking the highest-rated player.")
    print("=======================================================================")