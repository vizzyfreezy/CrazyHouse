import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm # A library for creating smart progress bars

# ==============================================================================
# SIMULATION CONFIGURATION - EDIT THIS SECTION
# ==============================================================================

# Define the list of players (must match names in your dataset, lowercase)
TOURNAMENT_PLAYERS = [
    'anthonyoja',
    'hardeywale',
    'zlater007',
    'bayormiller_cno',
    'warlock_dabee',
    'ezengwori',
    'ovokodigood',
    'ageless_2',
    'bb_thegame',
    # Add or remove players as needed
]

# Number of times to simulate the entire tournament
NUM_SIMULATIONS = 2000 # Higher is more accurate but slower. 1000-5000 is a good range.

# Estimated probability of a draw in a game. Crazyhouse draws are rare.
DRAW_PROBABILITY = 0.03 # 3% chance of a draw


# ==============================================================================
# CORE SIMULATION LOGIC
# ==============================================================================

def simulate_game(player1_stats, player2_stats):
    """
    Simulates a single game between two players based on their ratings.
    Returns the points scored by each player (p1_score, p2_score).
    """
    # Check for a draw first
    if np.random.rand() < DRAW_PROBABILITY:
        return 0.5, 0.5

    # If not a draw, calculate win probability using the Elo formula
    r1 = player1_stats['Current Rating']
    r2 = player2_stats['Current Rating']
    
    # Expected score for Player 1
    e1 = 1 / (1 + 10**((r2 - r1) / 400))
    
    # Win probability for Player 1, adjusted for the non-draw outcome
    win_prob_p1 = (e1 - 0.5 * DRAW_PROBABILITY) / (1 - DRAW_PROBABILITY)

    if np.random.rand() < win_prob_p1:
        return 1.0, 0.0 # Player 1 wins
    else:
        return 0.0, 1.0 # Player 2 wins

def run_single_tournament(players, player_data):
    """
    Runs one full round-robin tournament simulation.
    Each player plays every other player once.
    """
    # Initialize scoreboard for this tournament
    scores = pd.Series(0.0, index=players)
    
    # Generate all unique pairings
    for p1_name, p2_name in itertools.combinations(players, 2):
        p1_stats = player_data.loc[p1_name]
        p2_stats = player_data.loc[p2_name]
        
        p1_score, p2_score = simulate_game(p1_stats, p2_stats)
        
        scores[p1_name] += p1_score
        scores[p2_name] += p2_score
        
    return scores


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    try:
        # Load the comprehensive player data
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        print("--- Player data loaded successfully. ---")
    except FileNotFoundError:
        print("\n!!! ERROR: 'player_features.csv' not found.")
        print("Please run the 'engineering.py' script first to generate the data file.")
        exit()

    # Verify that all players in the tournament list exist in the data
    missing_players = [p for p in TOURNAMENT_PLAYERS if p not in player_data_df.index]
    if missing_players:
        print(f"\n!!! ERROR: The following players are in your tournament list but not in the data file: {missing_players}")
        print("Please check for typos or run engineering.py again.")
        exit()
        
    print(f"\n--- Starting Monte Carlo Simulation ---")
    print(f"  Tournament Participants: {len(TOURNAMENT_PLAYERS)}")
    print(f"  Number of Simulations: {NUM_SIMULATIONS}\n")

    # --- Run the main simulation loop ---
    all_results = []
    # tqdm is a library that creates a nice progress bar
    for _ in tqdm(range(NUM_SIMULATIONS), desc="Simulating Tournaments"):
        final_scores = run_single_tournament(TOURNAMENT_PLAYERS, player_data_df)
        all_results.append(final_scores)
        
    results_df = pd.DataFrame(all_results)
    
    print("\n--- Simulation Complete. Analyzing Results... ---")
    
    # --- Analyze and summarize the results ---
    summary = pd.DataFrame(index=TOURNAMENT_PLAYERS)
    
    # Calculate win percentages
    winners = results_df.idxmax(axis=1)
    summary['Win %'] = (winners.value_counts() / NUM_SIMULATIONS * 100).fillna(0)
    
    # Calculate Top 3 finish percentages
    top3_counts = pd.Series(0, index=TOURNAMENT_PLAYERS)
    for i in range(len(results_df)):
        top_finishers = results_df.iloc[i].nlargest(3).index
        top3_counts[top_finishers] += 1
    summary['Top 3 %'] = top3_counts / NUM_SIMULATIONS * 100
    
    # Calculate score statistics
    summary['Avg Score'] = results_df.mean(axis=0)
    summary['Std Dev Score'] = results_df.std(axis=0)
    summary['Max Score'] = results_df.max(axis=0)
    summary['Min Score'] = results_df.min(axis=0)

    # --- Display the final report ---
    print("\n\n=======================================================================")
    print("                TOURNAMENT SIMULATION PREDICTIONS")
    print("=======================================================================")
    print(f"Based on {NUM_SIMULATIONS} simulated round-robin tournaments.")
    
    pd.set_option('display.width', 100)
    print(summary.sort_values(by='Win %', ascending=False).round(1))
    print("=======================================================================")