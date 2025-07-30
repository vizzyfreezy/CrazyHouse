import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
import subprocess 
from simulate_teambattle import  build_h2h_data_from_games
from multiprocessing import Pool, cpu_count
cpu_count=8

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
WEIGHT_RATING = 0.50
WEIGHT_H2H = 0.45
WEIGHT_STYLE = 0.05
DRAW_PROBABILITY = 0.03
GAMES_PER_MATCHUP = 2

# ==============================================================================
# CORE SIMULATION & ANALYSIS LOGIC
# ==============================================================================

def calculate_h2h_advantage(p1_name, p2_name, h2h_data):
    p1_wins, p2_wins = h2h_data.get(p1_name, {}).get(p2_name, {}).get('wins', 0), h2h_data.get(p2_name, {}).get(p1_name, {}).get('wins', 0)
    total_games = p1_wins + p2_wins
    if total_games < 5: return 0.0
    return ((p1_wins / total_games) - 0.5) * 2

def calculate_style_advantage(p1_stats, p2_stats):
    speed_adv = p1_stats['Speed_norm'] - p2_stats['Speed_norm']
    consistency_adv = p1_stats['Consistency_norm'] - p2_stats['Consistency_norm']
    aggressiveness_adv = (p1_stats['Aggressiveness_norm'] - p2_stats['Aggressiveness_norm']) * 0.5
    return (speed_adv + consistency_adv + aggressiveness_adv) / 2.5

def simulate_game_advanced(p1_stats, p2_stats, h2h_data):
    if np.random.rand() < DRAW_PROBABILITY: return 0.5, 0.5
    base_prob_p1 = 1 / (1 + 10**((p2_stats['Current Rating'] - p1_stats['Current Rating']) / 400))
    h2h_adv = calculate_h2h_advantage(p1_stats.name, p2_stats.name, h2h_data)
    style_adv = calculate_style_advantage(p1_stats, p2_stats)
    matchup_score_p1 = (base_prob_p1 - 0.5) * WEIGHT_RATING + h2h_adv * WEIGHT_H2H + style_adv * WEIGHT_STYLE
    final_prob_p1 = np.clip(matchup_score_p1 + 0.5, 0.05, 0.95)
    return (1.0, 0.0) if np.random.rand() < final_prob_p1 else (0.0, 1.0)

def run_mini_simulation(my_team, opponent_team, player_data, h2h_data):
    if not opponent_team: return {p: 0 for p in my_team}
    all_player_results = []
    for _ in range(SIMULATIONS_PER_CANDIDATE):
        player_scores = {player: 0.0 for player in my_team + opponent_team}
        for p1_name, p2_name in itertools.product(my_team, opponent_team):
            for _ in range(GAMES_PER_MATCHUP):
                p1_score, p2_score = simulate_game_advanced(player_data.loc[p1_name], player_data.loc[p2_name], h2h_data)
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
    candidate, my_team, opponent_team, player_data, h2h_data = args
    test_my_team = my_team + [candidate]
    avg_scores = run_mini_simulation(test_my_team, opponent_team, player_data, h2h_data)
    return candidate, avg_scores.get(candidate, 0)

def find_best_pick_parallel(my_team, opponent_team, player_pool, player_data, h2h_data):
    """
    Analyzes the player pool in parallel to find the optimal pick.
    """
    global cpu_count
    # Prepare the arguments for each parallel task
    tasks = [(candidate, my_team, opponent_team, player_data, h2h_data) for candidate in player_pool]
    
    candidate_scores = {}
    
    # Create a pool of worker processes that uses all available CPU cores
    with Pool(processes=cpu_count) as p:
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
        
    h2h_data = build_h2h_data_from_games(games_df)
    
    while player_pool:
        display_draft_state(my_team, opponent_team, player_pool)
        choice = input("Whose turn is it? (1: My Turn, 2: Opponent's Turn, 'exit' to end): ").strip()
        
        if choice == '1':
            print("\n--- RUNNING ANALYSIS FOR YOUR PICK (IN PARALLEL) ---")
            recommendations = find_best_pick_parallel(my_team, opponent_team, player_pool, player_data_df, h2h_data)
            
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