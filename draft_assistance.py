import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
import subprocess 
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
# DRAFT ASSISTANT CONFIGURATION - EDIT THIS SECTION
# ==============================================================================
PLAYER_POOL = [
    'anthonyoja', 'zgm-giantkiller', 'bayormiller_cno', 'trappatoni', 'zlater007', 'b4elma',
    'patzernoblechuks', 'ovokodigood', 'adet2510', 'noblechuks_cno', 'spicypearl8', 'tommybrooks',
    'ezengwori', 'ageless_2', 'crazybugg', 'warlock_dabee', 'hardeywale', 'vegakov',
    'kirekachesschamp', 'bb_thegame', 'martins177', 'lexzero2', 'overgiftedlight', 'nevagivup'
]
SIMULATIONS_PER_CANDIDATE = 500
GAMES_PER_MATCHUP = 2

# ==============================================================================
# CORE SIMULATION & ANALYSIS LOGIC
# ==============================================================================

def run_mini_simulation(my_team, opponent_team, player_data, h2h_data):
    if not opponent_team: return {p: 0 for p in my_team}
    all_player_results = []
    for _ in range(SIMULATIONS_PER_CANDIDATE):
        player_scores = {player: 0.0 for player in my_team + opponent_team}
        for p1_name, p2_name in itertools.product(my_team, opponent_team):
            for _ in range(GAMES_PER_MATCHUP):
                # Pass player_data_dict and h2h_wins_dict directly to simulate_game_advanced
                p1_stats = player_data[p1_name] # Access as dictionary
                p2_stats = player_data[p2_name] # Access as dictionary
                p1_score, p2_score = simulate_game_advanced(p1_name, p2_name, p1_stats, p2_stats, h2h_data)
                player_scores[p1_name] += p1_score
                player_scores[p2_name] += p2_score
        all_player_results.append(pd.Series(player_scores))
    return pd.DataFrame(all_player_results).mean(axis=0)

# --- WRAPPER FUNCTION FOR PARALLEL PROCESSING ---
def analyze_candidate_wrapper(args):
    """
    A simple wrapper to allow the main analysis function to be called by the multiprocessing Pool.
    It unpacks the arguments and runs the simulation for a single candidate.
    """
    candidate, my_team, opponent_team, player_data_dict, h2h_wins_dict = args
    test_my_team = my_team + [candidate]
    avg_scores = run_mini_simulation(test_my_team, opponent_team, player_data_dict, h2h_wins_dict)
    return candidate, avg_scores.get(candidate, 0)

def find_best_pick_parallel(my_team, opponent_team, player_pool, player_data_dict, h2h_wins_dict):
    """
    Analyzes the player pool in parallel to find the optimal pick.
    """
    # Prepare the arguments for each parallel task
    tasks = [(candidate, my_team, opponent_team, player_data_dict, h2h_wins_dict) for candidate in player_pool]
    
    candidate_scores = {}
    
    # Use all available CPU cores
    num_processes = os.cpu_count() if os.cpu_count() else 4 # Fallback to 4 if not detectable
    print(f"Using {num_processes} CPU cores for parallel analysis.")

    with Pool(processes=num_processes) as p:
        # Use tqdm to create a progress bar for the parallel processing
        # p.imap_unordered processes tasks in parallel and yields results as they complete
        results = list(tqdm(p.imap_unordered(analyze_candidate_wrapper, tasks), total=len(tasks), desc="Analyzing Candidates (Parallel)"))
    
    # Collect results from the parallel tasks
    for candidate, score in results:
        candidate_scores[candidate] = score
        
    return sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)

def display_draft_state(my_team, opponent_team, player_pool):
    """Prints the current state of the draft."""
    print("\n" + "="*50)
    print("                      DRAFT STATE")
    print("="*50)
    print(f"My Team ({len(my_team)}): {', '.join(my_team) if my_team else 'None'}")
    print(f"Opponent's Team ({len(opponent_team)}): {', '.join(opponent_team) if opponent_team else 'None'}")
    print(f"Players Remaining ({len(player_pool)}): {', '.join(player_pool[:5])}...")
    print("="*50 + "\n")
##Interactive Selection
def fzf_select(options):
    """Display a terminal-based selection box using fzf."""
    try:
        result = subprocess.run(
            ['fzf'], input='\n'.join(options), text=True, capture_output=True
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print("fzf is not installed. Please run: pkg install fzf")
        exit()

def extract_player_name(line):
    """Extract player name from line that may contain scores."""
    return line.split()[0].lower()

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv')
        print("--- Player and game data loaded successfully. Database is ready. ---")
    except FileNotFoundError:
        print("\n!!! ERROR: 'player_features.csv' or 'all_games.csv' not found.")
        print("Please run the 'engineering.py' script first.")
        exit()

    player_pool = sorted([p.lower() for p in PLAYER_POOL])
    my_team, opponent_team = [], []
    
    missing_players = [p for p in player_pool if p not in player_data_df.index]
    if missing_players:
        print(f"\n!!! ERROR: The following players in your POOL are not in the data file: {missing_players}")
        exit()
        
    h2h_wins_dict = build_h2h_wins_dict_from_games(games_df)
    
    while player_pool:
        display_draft_state(my_team, opponent_team, player_pool)
        choice = input("Whose turn is it? (1: My Turn, 2: Opponent's Turn, 'exit' to end): ").strip()
        
        if choice == '1':
            print("\n--- RUNNING ANALYSIS FOR YOUR PICK (IN PARALLEL) ---")
            # Pass player_data_df as a dictionary to run_mini_simulation
            recommendations = find_best_pick_parallel(my_team, opponent_team, player_pool, player_data_df.to_dict('index'), h2h_wins_dict)
            
            # Build fzf options
            print("\n--- SELECT FROM RECOMMENDATIONS OR FULL POOL ---")
            fzf_options = [
                f"{player}  (score: {score:.2f})" for player, score in recommendations[:5]
            ] + ['────────────'] + player_pool
            
            selected_line = fzf_select(fzf_options)
            selected_player = extract_player_name(selected_line)

            if selected_player in player_pool:
                my_team.append(selected_player)
                player_pool.remove(selected_player)
                print(f"\n>> You have drafted {selected_player.upper()}.")
            else:
                print(">> Invalid pick or cancelled.")
        
        elif choice == '2':
            opponent_pick = fzf_select(player_pool)
            if opponent_pick in player_pool:
                opponent_team.append(opponent_pick)
                player_pool.remove(opponent_pick)
                print(f"\n>> Opponent has drafted {opponent_pick.upper()}.")
            else:
                print(">> Invalid pick or cancelled.")
        
        elif choice.lower() in ['exit', 'quit']:
            break
        else:
            print(">> Invalid choice. Please enter 1, 2, or 'exit'.")

    print("\n--- FINAL DRAFT RESULTS ---")
    display_draft_state(my_team, opponent_team, player_pool)
    print("Draft complete.")