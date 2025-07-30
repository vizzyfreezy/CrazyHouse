import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

# ==============================================================================
# ADVANCED SIMULATION CONFIGURATION - EDIT THIS SECTION
# ==============================================================================

TOURNAMENT_PLAYERS = [
    'anthonyoja', 'hardeywale', 'bayormiller_cno',
    'warlock_dabee', 'ezengwori', 'ovokodigood', 'ageless_2', 'bb_thegame','adet2510','vegakov'
]
NUM_SIMULATIONS = 2000
DRAW_PROBABILITY = 0.03

# --- Matchup Factor Weights ---
# These weights determine the importance of different factors. They should sum to 1.0.
WEIGHT_RATING = 0.30  # 60% of the prediction comes from the rating difference
WEIGHT_H2H = 0.60     # 25% comes from the direct Head-to-Head record
WEIGHT_STYLE = 0.10# 15% comes from the clash of player archetypes

# ==============================================================================
# CORE SIMULATION LOGIC (UPGRADED)
# ==============================================================================

def calculate_h2h_advantage(p1_name, p2_name, h2h_data):
    """
    Calculates H2H advantage on a scale from -1 (total domination by p2) to +1 (total domination by p1).
    Returns 0 if no significant H2H history exists.
    """
    p1_wins = h2h_data.get(p1_name, {}).get(p2_name, {}).get('wins', 0)
    p2_wins = h2h_data.get(p2_name, {}).get(p1_name, {}).get('wins', 0)
    total_games = p1_wins + p2_wins
    
    if total_games < 5: # Don't use H2H if there isn't enough data
        return 0.0
    
    win_pct_p1 = p1_wins / total_games
    # Scale the win percentage (0 to 1) to an advantage score (-1 to +1)
    return (win_pct_p1 - 0.5) * 2

def calculate_style_advantage(p1_stats, p2_stats):
    """
    Calculates a style advantage score based on archetype clashes.
    Scale is -1 to +1.
    """
    # Compare Speed: A fast player has an advantage over a slow one.
    speed_adv = p1_stats['Speed_norm'] - p2_stats['Speed_norm']
    
    # Compare Consistency: A consistent player has an advantage over a volatile one.
    consistency_adv = p1_stats['Consistency_norm'] - p2_stats['Consistency_norm']

    # Compare Aggressiveness: Assume aggressive players have a slight edge in forcing mistakes.
    aggressiveness_adv = (p1_stats['Aggressiveness_norm'] - p2_stats['Aggressiveness_norm']) * 0.5 # Less impact
    
    # Average the advantages. The max possible score is (1 + 1 + 0.5) / 3 = 0.83, so we normalize.
    total_adv = (speed_adv + consistency_adv + aggressiveness_adv) / 2.5
    return total_adv


def simulate_game_advanced(p1_stats, p2_stats, h2h_data):
    """
    Simulates a single game using a multi-factor prediction model.
    """
    if np.random.rand() < DRAW_PROBABILITY:
        return 0.5, 0.5

    # 1. Base Win Probability from Rating
    r1, r2 = p1_stats['Current Rating'], p2_stats['Current Rating']
    base_prob_p1 = 1 / (1 + 10**((r2 - r1) / 400))
    
    # 2. H2H Advantage
    h2h_adv = calculate_h2h_advantage(p1_stats.name, p2_stats.name, h2h_data)
    
    # 3. Style Advantage
    style_adv = calculate_style_advantage(p1_stats, p2_stats)
    
    # --- Combine Factors into a "Matchup Score" ---
    # The advantages modify the base probability. An advantage of +1 adds 1, -1 subtracts 1.
    # The weights control the magnitude of this modification.
    
    matchup_score_p1 = (base_prob_p1 - 0.5) + \
                       (h2h_adv * WEIGHT_H2H) + \
                       (style_adv * WEIGHT_STYLE)
                       
    # Convert the matchup score back to a 0-1 probability
    final_prob_p1 = matchup_score_p1 * (1 / (WEIGHT_RATING + WEIGHT_H2H + WEIGHT_STYLE)) + 0.5
    
    # Ensure probability is within a reasonable range [0.05, 0.95] to allow for upsets
    final_prob_p1 = np.clip(final_prob_p1, 0.05, 0.95)

    if np.random.rand() < final_prob_p1:
        return 1.0, 0.0
    else:
        return 0.0, 1.0

def run_single_tournament(players, player_data, h2h_data):
    """Runs one full round-robin tournament simulation."""
    scores = pd.Series(0.0, index=players)
    for p1_name, p2_name in itertools.combinations(players, 2):
        p1_stats, p2_stats = player_data.loc[p1_name], player_data.loc[p2_name]
        p1_score, p2_score = simulate_game_advanced(p1_stats, p2_stats, h2h_data)
        scores[p1_name] += p1_score
        scores[p2_name] += p2_score
    return scores

def build_h2h_data_from_games(games_df, qualified_players):
    """Builds the H2H dictionary needed for the simulation."""
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
    return h2h_data

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv') # Assuming you save games data for H2H
        print("--- Player and game data loaded successfully. ---")
    except FileNotFoundError:
        print("\n!!! ERROR: 'player_features.csv' or 'all_games.csv' not found.")
        print("Please run 'engineering.py' first.")
        # As a fallback, try to build H2H from the simulation's limited view
        # This part requires you to have the original games_df available.
        # For simplicity, let's assume engineering.py creates 'all_games.csv'
        # To do this, add `games_df.to_csv('all_games.csv')` to your engineering script.
        exit()

    missing_players = [p for p in TOURNAMENT_PLAYERS if p not in player_data_df.index]
    if missing_players:
        print(f"\n!!! ERROR: The following players are missing: {missing_players}")
        exit()
        
    # Build the necessary H2H data
    h2h_data = build_h2h_data_from_games(games_df, TOURNAMENT_PLAYERS)
        
    print(f"\n--- Starting Advanced Monte Carlo Simulation ---")
    print(f"Factors: Rating ({WEIGHT_RATING*100:.0f}%), H2H ({WEIGHT_H2H*100:.0f}%), Style ({WEIGHT_STYLE*100:.0f}%)\n")

    all_results = []
    for _ in tqdm(range(NUM_SIMULATIONS), desc="Simulating Tournaments"):
        final_scores = run_single_tournament(TOURNAMENT_PLAYERS, player_data_df, h2h_data)
        all_results.append(final_scores)
        
    results_df = pd.DataFrame(all_results)
    
    print("\n--- Simulation Complete. Analyzing Results... ---")
    
    summary = pd.DataFrame(index=TOURNAMENT_PLAYERS)
    summary['Win %'] = (results_df.idxmax(axis=1).value_counts() / NUM_SIMULATIONS * 100).fillna(0)
    top3_counts = pd.Series(0, index=TOURNAMENT_PLAYERS)
    for i in range(len(results_df)):
        top3_counts[results_df.iloc[i].nlargest(3).index] += 1
    summary['Top 3 %'] = top3_counts / NUM_SIMULATIONS * 100
    summary['Avg Score'] = results_df.mean(axis=0)
    summary['Std Dev Score'] = results_df.std(axis=0)

    print("\n\n=======================================================================")
    print("            ADVANCED TOURNAMENT SIMULATION PREDICTIONS")
    print("=======================================================================")
    print(f"Based on {NUM_SIMULATIONS} simulated round-robin tournaments.")
    print(summary.sort_values(by='Win %', ascending=False).round(1))
    print("=======================================================================")