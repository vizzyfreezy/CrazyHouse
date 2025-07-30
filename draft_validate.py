import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
from multiprocessing import Pool, cpu_count
cpu_count=8
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

# --- Matchup Factor Weights (Final Hybrid Model) ---
WEIGHT_RATING = 0.50
WEIGHT_H2H = 0.45
WEIGHT_STYLE = 0.05
DRAW_PROBABILITY = 0.03

# ==============================================================================
# CORE SIMULATION & DRAFT LOGIC (DO NOT EDIT)
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

def analyze_candidate_wrapper(args):
    candidate, my_team, opponent_team, player_data, h2h_data = args
    test_my_team = my_team + [candidate]
    avg_scores = run_mini_simulation(test_my_team, opponent_team, player_data, h2h_data)
    return candidate, avg_scores.get(candidate, 0)

def find_best_pick_parallel(my_team, opponent_team, player_pool, player_data, h2h_data):
    tasks = [(candidate, my_team, opponent_team, player_data, h2h_data) for candidate in player_pool]
    with Pool(processes=cpu_count) as p:
        results = list(tqdm(p.imap_unordered(analyze_candidate_wrapper, tasks), total=len(tasks), desc="Analyzing Pick", leave=False))
    return sorted(results, key=lambda item: item[1], reverse=True)[0][0]

def build_h2h_data_from_games(games_df):
    h2h_groups = games_df.groupby(['winner_name', 'loser_name']).size().reset_index(name='wins')
    h2h_data = {}
    all_players = pd.concat([games_df['winner_name'], games_df['loser_name']]).unique()
    for p1 in all_players:
        h2h_data[p1] = {}
        for p2 in all_players:
            if p1 == p2: continue
            wins = h2h_groups[(h2h_groups['winner_name'] == p1) & (h2h_groups['loser_name'] == p2)]['wins'].sum()
            losses = h2h_groups[(h2h_groups['winner_name'] == p2) & (h2h_groups['loser_name'] == p1)]['wins'].sum()
            h2h_data[p1][p2] = {'wins': wins, 'losses': losses}
    return h2h_data

def run_final_showdown_simulation(teams, player_data, h2h_data):
    team_scores = {team_name: 0.0 for team_name in teams}
    for p1_name, p2_name in itertools.product(teams['Smart Team'], teams['Random Team']):
        for _ in range(GAMES_PER_MATCHUP):
            p1_score, p2_score = simulate_game_advanced(player_data.loc[p1_name], player_data.loc[p2_name], h2h_data)
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

    player_pool = sorted([p.lower() for p in PLAYER_POOL])
    smart_team, random_team = [], []
    
    missing_players = [p for p in player_pool if p not in player_data_df.index]
    if missing_players:
        print(f"\n!!! ERROR: The following players in your POOL are not in the data file: {missing_players}")
        exit()
        
    h2h_data = build_h2h_data_from_games(games_df)
    
    # --- 1. Automated Draft ---
    print("\n--- Starting Automated Draft: Smart vs. Random ---")
    turn = 0
    draft_progress = tqdm(total=len(player_pool), desc="Drafting Players")
    while player_pool:
        if turn % 2 == 0: # Smart Team's turn
            best_pick = find_best_pick_parallel(smart_team, random_team, player_pool, player_data_df, h2h_data)
            smart_team.append(best_pick)
            player_pool.remove(best_pick)
        else: # Random Team's turn
            # To make it slightly more realistic, let's assume random picks the best remaining by rating
            # This is a tougher test than pure random.
            # random_pick = random.choice(player_pool)
            
            # Create a temporary DataFrame of players left in the pool
            pool_df = player_data_df.loc[player_pool]
            random_pick = pool_df['Current Rating'].idxmax() # Picks highest rated player left
            
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
        final_scores = run_final_showdown_simulation(final_teams, player_data_df, h2h_data)
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