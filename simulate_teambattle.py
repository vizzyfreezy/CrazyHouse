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
    "my_team": [
        "bayormiller_cno", "warlock_dabee", "crazybugg", "anthonyoja",
        "zlater007", "nevagivup", "noblechuks_cno", "adet2510",
        "trappatoni", "tommybrooks", "b4elma", "spicypearl8"
    ],
    "opponent_team": [
        "hardeywale", "zgm-giantkiller", "kirekachesschamp", "ageless_2",
        "ezengwori", "bb_thegame", "vegakov", "ovokodigood",
        "martins177", "patzernoblechuks", "lexzero2", "overgiftedlight"
    ]
}
TOURNAMENT_DURATION_MINUTES = 90
NUM_SIMULATIONS = 1000
TIME_CONTROL_SECONDS = 120
WEIGHT_RATING = 0.50
WEIGHT_H2H = 0.45
WEIGHT_STYLE = 0.05
DRAW_PROBABILITY = 0.0001

# ==============================================================================
# CORE SIMULATION LOGIC (ADVANCED)
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

def estimate_game_duration(base_time_seconds):
    min_duration, max_duration = 0.4 * base_time_seconds * 2, 0.9 * base_time_seconds * 2
    return random.uniform(min_duration, max_duration)

def find_pairings(player_states, teams):
    pairings, player_to_team = [], {player: name for name, players in teams.items() for player in players}
    waiting_players = [p for p, s in player_states.items() if s['status'] == 'waiting']
    random.shuffle(waiting_players)
    waiting_players.sort(key=lambda p: player_states[p]['score'], reverse=True)
    paired_players = set()
    for p1_name in waiting_players:
        if p1_name in paired_players: continue
        p1_team = player_to_team[p1_name]
        best_opponent = None
        for p2_name in waiting_players:
            if p2_name in paired_players or p1_name == p2_name: continue
            if player_to_team[p2_name] == p1_team: continue
            if p2_name in player_states[p1_name]['recent_opponents']: continue
            best_opponent = p2_name
            break
        if best_opponent:
            pairings.append((p1_name, best_opponent))
            paired_players.add(p1_name)
            paired_players.add(best_opponent)
    return pairings

def run_single_arena_simulation(teams, player_data, h2h_data, duration_seconds, time_control_seconds):
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

        p1_stats, p2_stats = player_data.loc[p1_finished], player_data.loc[p2_finished]
        p1_score, p2_score = simulate_game_advanced(p1_stats, p2_stats, h2h_data)
        
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

def build_h2h_data_from_games(games_df):
    # Step 1: Count wins per (winner, loser) pair
    h2h_pairs = games_df.groupby(['winner_name', 'loser_name']).size()
    
    # Step 2: Convert to a dictionary for fast lookup
    h2h_dict = h2h_pairs.to_dict()  # key = (winner_name, loser_name), value = win count

    # Step 3: Get all players
    all_players = pd.unique(games_df[['winner_name', 'loser_name']].values.ravel())

    # Step 4: Build H2H structure
    h2h_data = {}
    for p1 in all_players:
        h2h_data[p1] = {}
        for p2 in all_players:
            if p1 == p2: continue
            wins = h2h_dict.get((p1, p2), 0)
            losses = h2h_dict.get((p2, p1), 0)
            h2h_data[p1][p2] = {'wins': wins, 'losses': losses}
    
    return h2h_data
def run_simulation_wrapper(_):
    return run_single_arena_simulation(
        TEAMS,
        player_data_df,
        h2h_data,
        TOURNAMENT_DURATION_MINUTES * 60,
        TIME_CONTROL_SECONDS
    )
# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
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

    h2h_data = build_h2h_data_from_games(games_df)
    print(f"\n--- Starting Dynamic Arena Team Battle Simulation (with Score Sheets) ---")

    NUM_PROCESSES = 7

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(run_simulation_wrapper, range(NUM_SIMULATIONS)), total=NUM_SIMULATIONS))

    all_team_results, all_player_scores = [], []
    first_sim_details = None

    for i, (team_scores, player_details) in enumerate(results):
        all_team_results.append(team_scores)
        all_player_scores.append(player_details['score'])
        if i == 0:
            first_sim_details = player_details

    team_results_df = pd.DataFrame(all_team_results)
    player_scores_df = pd.DataFrame(all_player_scores)
    
    margin_records = []
    
    for i, result in enumerate(all_team_results):
        team_names = result.index.tolist()
        if len(team_names) != 2:
            continue
    
        team_a, team_b = team_names
        score_a, score_b = result[team_a], result[team_b]
        margin = abs(score_a - score_b)
        winner = team_a if score_a > score_b else team_b
    
        margin_records.append({
            'index': i,
            'winner': winner,
            'loser': team_b if winner == team_a else team_a,
            'margin': margin,
            'score_winner': max(score_a, score_b),
            'score_loser': min(score_a, score_b)
        })
    
    margins_df = pd.DataFrame(margin_records)
    
    for team in TEAMS.keys():
        biggest_win = margins_df[margins_df['winner'] == team].sort_values(by='margin', ascending=False).head(1)
        if not biggest_win.empty:
            row = biggest_win.iloc[0]
            print(f"\n🔹 Highest-margin win for {team}:")
            print(f"  Simulation #{row['index']} — {row['score_winner']} vs {row['score_loser']} (margin: {row['margin']})")
# -------------------------------------

    print("\n--- Simulation Complete. Analyzing Results... ---")

    # Analyze and display team results
    winners = []
    for i in range(len(team_results_df)):
        row = team_results_df.iloc[i]
        if (row == row.max()).sum() > 1:
            winners.append('Draw')
        else:
            winners.append(row.idxmax())
    win_percentages = (pd.Series(winners).value_counts() / NUM_SIMULATIONS * 100).fillna(0)

    print("\n\n=======================================================================")
    print("            DYNAMIC ARENA TEAM BATTLE PREDICTIONS")
    print("=======================================================================")
    print(f"Based on {NUM_SIMULATIONS} simulated {TOURNAMENT_DURATION_MINUTES}-minute Arena Team Battles.\n")
    print("--- Teams ---")
    for name, players in TEAMS.items():
        print(f"  {name}: {', '.join(players)}")

    report_df = pd.DataFrame({'Win %': win_percentages})
    report_df['Avg Team Score'] = team_results_df.mean(axis=0)
    print("\n--- Predicted Team Outcomes ---")
    print(report_df.sort_values(by='Win %', ascending=False).round(1).to_string())

    print("\n\n--- Player Contribution Report (Averages) ---")
    player_report = pd.DataFrame(index=all_tournament_players)
    player_report['Team'] = {p: name for name, players in TEAMS.items() for p in players}
    player_report['Avg Score'] = player_scores_df.mean(axis=0)
    print(player_report.sort_values(by=['Team', 'Avg Score'], ascending=[True, False]).round(1).to_string())

    print("\n\n--- In-Depth Look: Details from a Single Sample Simulation ---")
    print("This shows one possible outcome to verify the simulation's logic.")
    print("Score codes: 4=Streak Win, 2=Normal Win, 1=Draw, 0=Loss\n")

    if first_sim_details is not None:
        debug_report = pd.DataFrame()
        debug_report['Team'] = {p: name for name, players in TEAMS.items() for p in players}
        debug_report['Final Score'] = first_sim_details['score']
        debug_report['Simulated Sheet'] = first_sim_details['sheet']
        debug_report['Opponent Order'] = first_sim_details['opponent_list']

        pd.set_option('display.max_colwidth', 100)
        print(debug_report.sort_values(by='Final Score', ascending=False).to_string())

    print("=======================================================================")