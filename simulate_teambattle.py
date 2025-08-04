import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
from multiprocessing import Pool
# ==============================================================================
# ARENA TEAM BATTLE SIMULATION CONFIGURATION - EDIT THIS SECTION
# ==============================================================================
TEAMS = {
    "Team Miller": [
        "anthonyoja", "warlock_dabee", "ageless_2", "mini_verse", "zlater007", 
        "ovokodigood", "patzernoblechuks", "martins177", "b4elma", "overgiftedlight", 
        "genuine99", "bayormiller_cno"
    ],
    "Team Wale": [
        "zgm-giantkiller", "kirekachesschamp", "crazybugg", "ezengwori", "vegakov", 
        "bb_thegame", "eburu_sanmi", "tommybrooks", "prommizex", "lexzero2", "spicypearl8", "hardeywale"
    ]
}
TOURNAMENT_DURATION_MINUTES = 90
NUM_SIMULATIONS = 1000
TIME_CONTROL_SECONDS = 120
WEIGHT_RATING = 0.50
WEIGHT_H2H = 0.30
WEIGHT_STYLE = 0.20
DRAW_PROBABILITY = 0.00001 # Very low, as per Crazyhouse game characteristics
INFAMY_RATING_THRESHOLD = 2200 # Define threshold for "infamous" players

# Clipping ranges for probabilities
DEFAULT_CLIP_RANGE = (0.05, 0.95)
INFAMOUS_CLIP_RANGE_TIGHT = (0.35, 0.65) # Tighter for infamous players with a moderate rating gap
INFAMOUS_CLIP_RANGE_VERY_TIGHT = (0.45, 0.55) # Even tighter for infamous players with a very small rating gap

# Rating gap thresholds for infamous players
RATING_GAP_THRESHOLD_1 = 50 # e.g., if gap < 50, use TIGHT
RATING_GAP_THRESHOLD_2 = 20 # e.g., if gap < 20, use VERY_TIGHT

# ==============================================================================
# CORE SIMULATION LOGIC (ADVANCED)
# ==============================================================================

def calculate_h2h_advantage(p1_name, p2_name, h2h_wins_dict):
    p1_wins = h2h_wins_dict.get((p1_name, p2_name), 0)
    p2_wins = h2h_wins_dict.get((p2_name, p1_name), 0)
    total_games = p1_wins + p2_wins
    if total_games < 10: return 0.0
    return ((p1_wins / total_games) - 0.5) * 2

def calculate_style_advantage(p1_stats, p2_stats):
    speed_adv = p1_stats['Speed_norm'] - p2_stats['Speed_norm']
    consistency_adv = p1_stats['Consistency_norm'] - p2_stats['Consistency_norm']
    aggressiveness_adv = (p1_stats['Aggressiveness_norm'] - p2_stats['Aggressiveness_norm']) * 0.5
    return (speed_adv + consistency_adv + aggressiveness_adv) / 2.5

def simulate_game_advanced(p1_name, p2_name, p1_stats, p2_stats, h2h_data):
    if np.random.rand() < DRAW_PROBABILITY: return 0.5, 0.5
    base_prob_p1 = 1 / (1 + 10**((p2_stats['Current Rating'] - p1_stats['Current Rating']) / 400))
    h2h_adv = calculate_h2h_advantage(p1_name, p2_name, h2h_data)
    style_adv = calculate_style_advantage(p1_stats, p2_stats)
    matchup_score_p1 = (base_prob_p1 - 0.5) * WEIGHT_RATING + h2h_adv * WEIGHT_H2H + style_adv * WEIGHT_STYLE

    # Dynamic clipping based on infamy and rating gap
    lower_clip, upper_clip = DEFAULT_CLIP_RANGE

    if p1_stats['Current Rating'] >= INFAMY_RATING_THRESHOLD and \
       p2_stats['Current Rating'] >= INFAMY_RATING_THRESHOLD:
        rating_gap = abs(p1_stats['Current Rating'] - p2_stats['Current Rating'])
        if rating_gap < RATING_GAP_THRESHOLD_2:
            lower_clip, upper_clip = INFAMOUS_CLIP_RANGE_VERY_TIGHT
        elif rating_gap < RATING_GAP_THRESHOLD_1:
            lower_clip, upper_clip = INFAMOUS_CLIP_RANGE_TIGHT

    final_prob_p1 = np.clip(matchup_score_p1 + 0.5, lower_clip, upper_clip)
    return (1.0, 0.0) if np.random.rand() < final_prob_p1 else (0.0, 1.0)

def estimate_game_duration(base_time_seconds):
    min_duration, max_duration = 0.4 * base_time_seconds * 2, 0.9 * base_time_seconds * 2
    return random.uniform(min_duration, max_duration)

def find_pairings(player_states, teams):
    player_to_team = {player: name for name, players in teams.items() for player in players}
    
    # Separate players by team and sort by score
    team_waiting_players = {team: sorted([p for p in players if player_states[p]['status'] == 'waiting'], 
                                        key=lambda p: player_states[p]['score'], reverse=True) 
                          for team, players in teams.items()}

    pairings, paired_players = [], set()
    
    # Assume two teams for pairing logic
    team1_name, team2_name = list(teams.keys())
    team1_players = team_waiting_players[team1_name]
    team2_players = team_waiting_players[team2_name]
    
    # Iterate through the smaller team and find opponents in the larger team
    if len(team1_players) > len(team2_players):
        team1_players, team2_players = team2_players, team1_players

    for p1 in team1_players:
        if p1 in paired_players: continue
        for p2 in team2_players:
            if p2 in paired_players: continue
            if p2 not in player_states[p1]['recent_opponents']:
                pairings.append((p1, p2))
                paired_players.add(p1)
                paired_players.add(p2)
                break
            
    return pairings

def run_single_arena_simulation(teams, player_data_dict, h2h_data, duration_seconds, time_control_seconds):
    """Runs one full, dynamic Arena, returning detailed player performance stats."""
    all_players = [p for team_list in teams.values() for p in team_list]
    player_states = {p: {'score': 0, 'status': 'waiting', 'recent_opponents': [], 'streak': 0,
                         'sheet': '', 'opponent_list': []} for p in all_players}
    current_time, active_games = 0, {}

    while current_time < duration_seconds:
        new_pairings = find_pairings(player_states, teams)
        for p1_name, p2_name in new_pairings:
            finish_time = current_time + estimate_game_duration(time_control_seconds)
            player_states[p1_name]['status'], player_states[p2_name]['status'] = 'playing', 'playing'
            active_games[(p1_name, p2_name)] = finish_time
    
        if not active_games: break
        
        (p1_finished, p2_finished), next_finish_time = min(active_games.items(), key=lambda item: item[1])
        current_time = next_finish_time
        
        if current_time >= duration_seconds: break

        p1_stats, p2_stats = player_data_dict[p1_finished], player_data_dict[p2_finished]
        p1_score, p2_score = simulate_game_advanced(p1_finished, p2_finished, p1_stats, p2_stats, h2h_data)
        
        # --- SCORE SHEET AND OPPONENT TRACKING ---
        player_states[p1_finished]['opponent_list'].append(p2_finished)
        player_states[p2_finished]['opponent_list'].append(p1_finished)
        
        if p1_score == 1:
            player_states[p1_finished]['streak'] += 1
            points_to_add = 2 if player_states[p1_finished]['streak'] < 3 else 4
            player_states[p1_finished]['sheet'] += str(points_to_add)
        elif p1_score == 0.5:
            player_states[p1_finished]['streak'] = 0
            points_to_add = 1
            player_states[p1_finished]['sheet'] += '1'
        else:
            player_states[p1_finished]['streak'] = 0
            points_to_add = 0
            player_states[p1_finished]['sheet'] += '0'
        player_states[p1_finished]['score'] += points_to_add
        
        if p2_score == 1:
            player_states[p2_finished]['streak'] += 1
            points_to_add = 2 if player_states[p2_finished]['streak'] < 3 else 4
            player_states[p2_finished]['sheet'] += str(points_to_add)
        elif p2_score == 0.5:
            player_states[p2_finished]['streak'] = 0
            points_to_add = 1
            player_states[p2_finished]['sheet'] += '1'
        else:
            player_states[p2_finished]['streak'] = 0
            points_to_add = 0
            player_states[p2_finished]['sheet'] += '0'
        player_states[p2_finished]['score'] += points_to_add

        player_states[p1_finished]['status'], player_states[p2_finished]['status'] = 'waiting', 'waiting'
        player_states[p1_finished]['recent_opponents'] = [p2_finished] + player_states[p1_finished]['recent_opponents'][:1]
        player_states[p2_finished]['recent_opponents'] = [p1_finished] + player_states[p2_finished]['recent_opponents'][:1]
        del active_games[(p1_finished, p2_finished)]

    final_player_details = pd.DataFrame(player_states).T
    final_team_scores = {name: 0 for name in teams}
    player_to_team = {player: name for name, players in teams.items() for player in players}
    for p_name, p_score in final_player_details['score'].items():
        final_team_scores[player_to_team[p_name]] += p_score

    return pd.Series(final_team_scores), final_player_details

def build_h2h_wins_dict_from_games(games_df):
    """Creates a dictionary mapping (winner, loser) to win counts for fast H2H lookup."""
    return games_df.groupby(['winner_name', 'loser_name']).size().to_dict()
def run_simulation_wrapper(_):
    return run_single_arena_simulation(
        TEAMS,
        player_data_dict, 
        h2h_wins_dict, # Pass the new H2H dictionary
        TOURNAMENT_DURATION_MINUTES * 60,
        TIME_CONTROL_SECONDS
    )
# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
import os

def analyze_and_display_results(team_results_df, player_scores_df, player_data_df, teams, num_simulations, duration_minutes):
    """Analyzes simulation results and prints a clean, insightful report."""
    
    # 1. Calculate Win Percentages
    winners = []
    for i in range(len(team_results_df)):
        row = team_results_df.iloc[i]
        if (row == row.max()).sum() > 1:
            winners.append('Draw')
        else:
            winners.append(row.idxmax())
    win_percentages = (pd.Series(winners).value_counts() / num_simulations * 100).fillna(0)

    # 2. Create Player Performance Report
    all_players = [p for team_list in teams.values() for p in team_list]
    player_report = pd.DataFrame(index=all_players)
    player_report['Team'] = {p: name for name, players in teams.items() for p in players}
    player_report['Avg Score'] = player_scores_df.mean(axis=0)
    player_report['Rating'] = player_report.index.map(player_data_df['Current Rating'])

    # 3. Identify High Performers and Weak Links
    for team_name in teams.keys():
        team_df = player_report[player_report['Team'] == team_name].copy()
        team_df['Rating Rank'] = team_df['Rating'].rank(ascending=False, method='min')
        team_df['Performance Rank'] = team_df['Avg Score'].rank(ascending=False, method='min')
        team_df['Rank Delta'] = team_df['Rating Rank'] - team_df['Performance Rank']
        
        player_report.loc[team_df.index, 'Performance Rank'] = team_df['Performance Rank']
        player_report.loc[team_df.index, 'Rank Delta'] = team_df['Rank Delta']

    # 4. Print the new, clean report
    print("\n\n=======================================================================")
    print("            DYNAMIC ARENA TEAM BATTLE PREDICTIONS")
    print("=======================================================================")
    print(f"Based on {num_simulations} simulated {duration_minutes}-minute Arena Team Battles.\n")
    
    team_report_df = pd.DataFrame({'Win %': win_percentages})
    team_report_df['Avg Team Score'] = team_results_df.mean(axis=0)
    print("--- Predicted Team Outcomes ---")
    print(team_report_df.sort_values(by='Win %', ascending=False).round(1).to_string())
    print("\n" + "-"*71)

    print("\n--- Player Performance Analysis ---")
    print("  'Rank Delta' shows performance vs. rating expectation.")
    print("  A positive delta means the player performed better than their rating implies.")
    print("  A negative delta means the player underperformed.\n")

    pd.set_option('display.max_rows', 100)
    for team_name in teams.keys():
        print(f"--- {team_name} ---")
        team_view = player_report[player_report['Team'] == team_name].copy()
        team_view['Rank Delta'] = team_view['Rank Delta'].apply(lambda x: f"+{x}" if x > 0 else str(x))
        print(team_view[['Avg Score', 'Performance Rank', 'Rating', 'Rank Delta']].sort_values(by='Avg Score', ascending=False).round(1).to_string())
        print("")

    print("=======================================================================")

if __name__ == "__main__":

    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: CSV files not found. Please run 'engineering.py' first.")
        exit()

    all_tournament_players = [player for team_players in TEAMS.values() for player in team_players]
    missing_players = [p for p in all_tournament_players if p not in player_data_df.index]
    if missing_players:
        print(f"\n!!! ERROR: The following players are missing: {missing_players}")
        exit()

    h2h_wins_dict = build_h2h_wins_dict_from_games(games_df)
    player_data_dict = player_data_df.to_dict('index')

    print(f"\n--- Starting Dynamic Arena Team Battle Simulation ---")

    try:
        NUM_PROCESSES = os.cpu_count()
        print(f"Using {NUM_PROCESSES} CPU cores for parallel processing.")
    except NotImplementedError:
        NUM_PROCESSES = 4 # Fallback value
        print("Could not determine number of CPUs, falling back to 4.")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(run_simulation_wrapper, range(NUM_SIMULATIONS)), total=NUM_SIMULATIONS))

    all_team_results, all_player_scores = [], []
    for team_scores, player_details in results:
        all_team_results.append(team_scores)
        all_player_scores.append(player_details['score'])

    team_results_df = pd.DataFrame(all_team_results)
    player_scores_df = pd.DataFrame(all_player_scores)
    
    analyze_and_display_results(team_results_df, player_scores_df, player_data_df, TEAMS, NUM_SIMULATIONS, TOURNAMENT_DURATION_MINUTES)
