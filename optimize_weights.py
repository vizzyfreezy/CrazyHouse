import subprocess
import itertools
import re
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd

# Import the refactored calibration function
from weight_calibrator import calculate_calibration_metrics, build_h2h_wins_dict_from_games

def run_calibration_wrapper(args):
    # This wrapper is needed because Pool.starmap expects a single iterable argument
    # and calculate_calibration_metrics expects multiple arguments.
    rating_weight, h2h_weight, style_weight, player_data_dict, h2h_wins_dict = args
    return calculate_calibration_metrics(player_data_dict, h2h_wins_dict, rating_weight, h2h_weight, style_weight)

def brute_force_grid_search():
    best_weights = None
    lowest_mse = float('inf')

    # Load data once
    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: CSV files not found. Please ensure 'player_features.csv' and 'all_games.csv' are in the current directory.")
        return None, float('inf')

    player_data_dict = player_data_df.to_dict('index')
    h2h_wins_dict = build_h2h_wins_dict_from_games(games_df)

    # Search in steps of 0.1 for now; can reduce to 0.05 later
    weight_range = np.arange(0.0, 1.1, 0.1)

    all_combinations_with_data = []
    for rating_weight, h2h_weight, style_weight in itertools.product(weight_range, repeat=3):
        # Normalize weights to sum to 1.0
        current_sum = rating_weight + h2h_weight + style_weight
        if current_sum == 0: # Avoid division by zero
            continue
        
        norm_r_w = rating_weight / current_sum
        norm_h_w = h2h_weight / current_sum
        norm_s_w = style_weight / current_sum

        # Ensure the sum is close to 1.0 after normalization
        if not np.isclose(norm_r_w + norm_h_w + norm_s_w, 1.0):
            continue
        
        all_combinations_with_data.append((norm_r_w, norm_h_w, norm_s_w, player_data_dict, h2h_wins_dict))

    print(f"Starting parallel grid search with {len(all_combinations_with_data)} combinations...")
    
    # Use all available CPU cores
    num_processes = os.cpu_count() if os.cpu_count() else 4 # Fallback to 4 if not detectable
    print(f"Using {num_processes} CPU cores.")

    with Pool(processes=num_processes) as pool:
        # Use tqdm for a progress bar
        for i, mse in enumerate(tqdm(pool.imap(run_calibration_wrapper, all_combinations_with_data), total=len(all_combinations_with_data), desc="Evaluating Weights")):
            current_weights = all_combinations_with_data[i][:3] # Extract only weights

            if np.isfinite(mse) and mse < lowest_mse:
                lowest_mse = mse
                best_weights = current_weights

    if best_weights:
        print(f"\n✅ Best Weights: Rating={best_weights[0]:.2f}, H2H={best_weights[1]:.2f}, Style={best_weights[2]:.2f}")
        print(f"🔍 Lowest MSE: {lowest_mse:.4f}")
    else:
        print("❌ No valid weight combination found.")

    return best_weights, lowest_mse

if __name__ == "__main__":
    best_weights, lowest_mse = brute_force_grid_search()

