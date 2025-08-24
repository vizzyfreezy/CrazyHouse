import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from simulate_teambattle import run_single_arena_simulation, build_h2h_wins_dict_from_games, TEAMS

# --- CONFIGURATION ---
RATING_WEIGHT_RANGE = (0.0, 1.0, 11)
H2H_WEIGHT_RANGE = (0.0, 1.0, 11)
STYLE_WEIGHT_RANGE = (0.0, 1.0, 11)
NUM_SIMULATIONS_PER_TOURNAMENT = 50 # Reduced for faster optimization
TOURNAMENT_DURATION_MINUTES = 90
TIME_CONTROL_SECONDS = 120
NUM_PROCESSES = cpu_count()

# --- TOURNAMENT DATA ---
HISTORICAL_TOURNAMENTS = {
    "2025-08-03": {
        "file": "/data/data/com.termux/files/home/Projects/CrazyHouse/tournament_results/lichess_tournament_2025.08.03_zMEUOhOb_cno-crazyhouse-team-battle.csv",
        "teams": {
            "millers-crib-": ["warlock_dabee", "ovokodigood", "ageless_2", "bayormiller_cno", "mini_verse", "zlater007", "anthonyoja", "patzernoblechuks", "overgiftedlight", "martins177", "genuine99", "b4elma"],
            "team-hardewale": ["kirekachesschamp", "hardeywale", "crazybugg", "zgm-giantkiller", "ezengwori", "eburu_sanmi", "vegakov", "mabunmi_cno", "tommybrooks", "spicypearl8", "bb_thegame", "prommizex"]
        }
    },
    "2025-07-20": {
        "file": "/data/data/com.termux/files/home/Projects/CrazyHouse/tournament_results/lichess_tournament_2025.07.20_y65zvK9W_cno-crazyhouse-team-battle.csv",
        "teams": {
            "millers-crib-": ["bayormiller_cno", "anthonyoja", "zlater007", "ovokodigood", "adet2510", "vegakov", "prommizex"],
            "team-hardewale": ["hardeywale", "mini_verse", "kirekachesschamp", "ageless_2", "clintonsalako", "patzernoblechuks", "bb_thegame"]
        }
    },
    "2025-07-27": {
        "file": "/data/data/com.termux/files/home/Projects/CrazyHouse/tournament_results/lichess_tournament_2025.07.27_ATjOdj4d_cno-crazyhouse-team-battle.csv",
        "teams": {
            "millers-crib-": ["anthonyoja", "zgm-giantkiller", "bayormiller_cno", "trappatoni", "zlater007", "b4elma", "patzernoblechuks", "ovokodigood", "adet2510", "talldove", "spicypearl8", "tommybrooks"],
            "team-hardewale": ["ezengwori", "ageless_2", "crazybugg", "warlock_dabee", "hardeywale", "vegakov", "kirekachesschamp", "bb_thegame", "martins177", "prommizex", "overgiftedlight", "clintonsalako"]
        }
    }
}

def get_actual_scores(tournament_file):
    df = pd.read_csv(tournament_file)
    # The team names in the CSV have a trailing hyphen, which we remove
    df['Team'] = df['Team'].str.rstrip('-')
    return df.groupby('Team')['Score'].sum().to_dict()

def generate_weight_combinations():
    w1_steps = np.linspace(*RATING_WEIGHT_RANGE)
    w2_steps = np.linspace(*H2H_WEIGHT_RANGE)
    w3_steps = np.linspace(*STYLE_WEIGHT_RANGE)
    all_combinations = []
    for r_w in w1_steps:
        for h_w in w2_steps:
            for s_w in w3_steps:
                if np.isclose(r_w + h_w + s_w, 1.0):
                    all_combinations.append((r_w, h_w, s_w))
    return all_combinations

def simulate_tournament_worker(args):
    tournament_name, teams, weights = args
    # This is a simplified simulation run for one tournament
    # It doesn't use multiprocessing itself, but is run in a pool
    team_results, _ = run_single_arena_simulation(
        teams,
        player_data_dict,
        h2h_wins_dict,
        TOURNAMENT_DURATION_MINUTES * 60,
        TIME_CONTROL_SECONDS,
        top_n_players=None
    )
    return tournament_name, team_results

def calculate_error_for_weights(weights):
    total_error = 0
    for name, data in HISTORICAL_TOURNAMENTS.items():
        actual_scores = get_actual_scores(data["file"])
        
        # Run multiple simulations to get average simulated scores
        simulated_results_list = []
        for _ in range(NUM_SIMULATIONS_PER_TOURNAMENT):
            team_results, _ = run_single_arena_simulation(
                data["teams"],
                player_data_dict,
                h2h_wins_dict,
                TOURNAMENT_DURATION_MINUTES * 60,
                TIME_CONTROL_SECONDS,
                top_n_players=None
            )
            simulated_results_list.append(team_results)
        
        # Average the simulated scores
        avg_simulated_scores = pd.DataFrame(simulated_results_list).mean().to_dict()

        # Determine actual winner and scores
        actual_winner = max(actual_scores, key=actual_scores.get)
        actual_loser = min(actual_scores, key=actual_scores.get) # Assuming 2 teams
        actual_margin = actual_scores[actual_winner] - actual_scores[actual_loser]

        # Determine simulated winner and scores
        simulated_winner = max(avg_simulated_scores, key=avg_simulated_scores.get)
        simulated_loser = min(avg_simulated_scores, key=avg_simulated_scores.get)
        simulated_margin = avg_simulated_scores[simulated_winner] - avg_simulated_scores[simulated_loser]

        # Error Component 1: Squared difference of winning margins
        total_error += (simulated_margin - actual_margin) ** 2

        # Error Component 2: Penalty for incorrect winner prediction
        if simulated_winner != actual_winner:
            total_error += 100000 # Large penalty for getting the winner wrong

        # Error Component 3: Squared difference of individual team scores (scaled)
        # This helps to get the overall score magnitudes closer
        for team, actual_score in actual_scores.items():
            simulated_score = avg_simulated_scores.get(team, 0)
            total_error += ((simulated_score - actual_score) / 10) ** 2 # Divide by 10 to reduce its impact relative to margin/winner
            
    return total_error

def find_best_weights_worker(weights):
    error = calculate_error_for_weights(weights)
    return weights, error

def init_worker(player_data, h2h_data):
    global player_data_dict
    global h2h_wins_dict
    player_data_dict = player_data
    h2h_wins_dict = h2h_data

if __name__ == "__main__":
    print("--- Tournament-Based Weight Optimization Engine ---")

    print("Step 1: Loading player features and historical game data...")
    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: CSV files not found. Please run 'engineering.py' first.")
        exit()

    player_data_dict_main = player_data_df.to_dict('index')
    h2h_wins_dict_main = build_h2h_wins_dict_from_games(games_df)

    print("Step 2: Generating weight combinations...")
    weight_combinations = generate_weight_combinations()
    print(f"Generated {len(weight_combinations)} valid combinations to test.")

    if not weight_combinations:
        print("!!! ERROR: No valid weight combinations generated. Check your RANGE settings.")
        exit()

    print(f"Step 3: Starting optimization across {NUM_PROCESSES} CPU cores...")
    best_error = float('inf')
    best_weights = None

    with Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(player_data_dict_main, h2h_wins_dict_main)) as pool:
        results = list(tqdm(pool.imap(find_best_weights_worker, weight_combinations), total=len(weight_combinations)))

    print("\nStep 4: Analyzing results...")
    for weights, error in results:
        if error < best_error:
            best_error = error
            best_weights = weights

    if best_weights:
        print("\n===================================================================")
        print("            OPTIMAL SIMULATION WEIGHTS FOUND")
        print("===================================================================")
        print(f"Lowest Error Score: {best_error:.2f}\n")
        print(f"Optimal Weights:")
        print(f"  - Rating: {best_weights[0]:.3f}")
        print(f"  - H2H:    {best_weights[1]:.3f}")
        print(f"  - Style:  {best_weights[2]:.3f}")
        print("\n--- Recommendation ---")
        print("Update the weights in 'simulate_teambattle.py' with these values.")
        print("===================================================================")

        best_weights_df = pd.DataFrame({
            'Factor': ['Rating', 'H2H', 'Style'],
            'Weight': [best_weights[0], best_weights[1], best_weights[2]]
        })
        best_weights_df.to_csv('optimal_tournament_weights.csv', index=False)
        print("\nOptimal weights saved to 'optimal_tournament_weights.csv'.")
    else:
        print("\n!!! Could not determine optimal weights.")
