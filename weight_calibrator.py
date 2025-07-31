import pandas as pd
import numpy as np
import argparse

# Import core simulation logic and constants from simulate_teambattle.py
from simulate_teambattle import (
    calculate_h2h_advantage,
    calculate_style_advantage,
    build_h2h_wins_dict_from_games,
    DRAW_PROBABILITY, # Import DRAW_PROBABILITY
    INFAMY_RATING_THRESHOLD, DEFAULT_CLIP_RANGE,
    INFAMOUS_CLIP_RANGE_TIGHT, INFAMOUS_CLIP_RANGE_VERY_TIGHT,
    RATING_GAP_THRESHOLD_1, RATING_GAP_THRESHOLD_2
)

def _get_outcome_probabilities(p1_name, p2_name, p1_stats, p2_stats, h2h_wins_dict,
                                rating_weight, h2h_weight, style_weight):
    """Calculates the probabilities of p1 winning, p2 winning, and a draw.
    This is a modified version of simulate_game_advanced to return probabilities.
    """
    base_prob_p1 = 1 / (1 + 10**((p2_stats['Current Rating'] - p1_stats['Current Rating']) / 400))
    h2h_adv = calculate_h2h_advantage(p1_name, p2_name, h2h_wins_dict)
    style_adv = calculate_style_advantage(p1_stats, p2_stats)

    matchup_score_p1 = (base_prob_p1 - 0.5) * rating_weight + \
                       h2h_adv * h2h_weight + \
                       style_adv * style_weight

    # Dynamic clipping based on infamy and rating gap (replicated from simulate_teambattle.py)
    lower_clip, upper_clip = DEFAULT_CLIP_RANGE
    if p1_stats['Current Rating'] >= INFAMY_RATING_THRESHOLD and \
       p2_stats['Current Rating'] >= INFAMY_RATING_THRESHOLD:
        rating_gap = abs(p1_stats['Current Rating'] - p2_stats['Current Rating'])
        if rating_gap < RATING_GAP_THRESHOLD_2:
            lower_clip, upper_clip = INFAMOUS_CLIP_RANGE_VERY_TIGHT
        elif rating_gap < RATING_GAP_THRESHOLD_1:
            lower_clip, upper_clip = INFAMOUS_CLIP_RANGE_TIGHT

    final_prob_p1_given_no_draw = np.clip(matchup_score_p1 + 0.5, lower_clip, upper_clip)

    # Factor in the global DRAW_PROBABILITY
    prob_p1_win = final_prob_p1_given_no_draw * (1 - DRAW_PROBABILITY)
    prob_p2_win = (1 - final_prob_p1_given_no_draw) * (1 - DRAW_PROBABILITY)
    prob_draw = DRAW_PROBABILITY

    # Ensure probabilities sum to 1 (due to floating point, might be slightly off)
    total_prob = prob_p1_win + prob_p2_win + prob_draw
    
    if total_prob == 0:
        # If total_prob is 0, it means all individual probabilities were 0.
        # This should ideally not happen with current logic, but as a safeguard,
        # return equal probabilities to avoid division by zero.
        return 1/3, 1/3, 1/3
    
    prob_p1_win /= total_prob
    prob_p2_win /= total_prob
    prob_draw /= total_prob

    return prob_p1_win, prob_p2_win, prob_draw

def calculate_calibration_metrics(player_data_dict, h2h_wins_dict, rating_weight, h2h_weight, style_weight):
    total_squared_error = 0
    game_count = 0
    correct_predictions = 0

    # Iterate through each game in all_games.csv
    # Note: games_df is not passed directly, assuming it's loaded externally
    # We need to iterate through the raw games_df data here.
    # For this refactor, we'll assume games_df is accessible or passed.
    # Let's adjust the signature to accept games_df as well.

    # This function will now be called from optimize_weights.py
    # So, games_df should be passed as an argument.
    # Let's assume games_df is passed as an argument to this function.

    # Re-reading games_df here for the purpose of this function's scope
    # In the final implementation, games_df will be passed from optimize_weights.py
    try:
        games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        # This should ideally not happen if optimize_weights.py loads it first
        return 1.0 # Return high MSE if data not found

    for index, game in games_df.iterrows():
        winner_name = game['winner_name']
        loser_name = game['loser_name']

        # Skip if player data is missing
        if winner_name not in player_data_dict or loser_name not in player_data_dict:
            continue

        p1_name = winner_name # The player who actually won
        p2_name = loser_name  # The player who actually lost

        p1_stats = player_data_dict[p1_name]
        p2_stats = player_data_dict[p2_name]

        # Get predicted probabilities for this matchup
        prob_p1_win, prob_p2_win, prob_draw = _get_outcome_probabilities(
            p1_name, p2_name, p1_stats, p2_stats, h2h_wins_dict,
            rating_weight, h2h_weight, style_weight
        )

        # --- Calculate MSE (for the actual winner) ---
        # Actual outcome for p1 (winner) is 1.0
        actual_outcome_p1 = 1.0
        squared_error = (actual_outcome_p1 - prob_p1_win)**2
        total_squared_error += squared_error

        # --- Calculate Prediction Accuracy ---
        # Determine predicted winner based on highest probability
        predicted_outcome = None
        if prob_p1_win > prob_p2_win and prob_p1_win > prob_draw:
            predicted_outcome = p1_name
        elif prob_p2_win > prob_p1_win and prob_p2_win > prob_draw:
            predicted_outcome = p2_name
        else:
            predicted_outcome = "Draw" # If draw prob is highest or tied

        # Check if prediction matches actual outcome
        # Since all_games.csv only has win/loss, a predicted draw is considered incorrect
        if predicted_outcome == p1_name: # If we predicted the actual winner
            correct_predictions += 1

        game_count += 1

    if game_count > 0:
        mean_squared_error = total_squared_error / game_count
        accuracy_percentage = (correct_predictions / game_count) * 100
        # print(f"Mean Squared Error: {mean_squared_error}")
        # print(f"Prediction Accuracy: {accuracy_percentage:.2f}%")
        return mean_squared_error
    else:
        # print("No games processed. Check your all_games.csv and player data.")
        return 1.0 # Return a high, but finite, MSE if no games processed

if __name__ == "__main__":
    # This block is now primarily for testing or direct execution if needed
    # It will not be used when called from optimize_weights.py
    parser = argparse.ArgumentParser(description="Calibrate simulation weights.")
    parser.add_argument("--rating_weight", type=float, required=True,
                        help="Weight for rating difference.")
    parser.add_argument("--h2h_weight", type=float, required=True,
                        help="Weight for head-to-head advantage.")
    parser.add_argument("--style_weight", type=float, required=True,
                        help="Weight for style advantage.")
    args = parser.parse_args()

    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: CSV files not found. Please ensure 'player_features.csv' and 'all_games.csv' are in the current directory.")
        exit()

    player_data_dict = player_data_df.to_dict('index')
    h2h_wins_dict = build_h2h_wins_dict_from_games(games_df)

    mse = calculate_calibration_metrics(player_data_dict, h2h_wins_dict, args.rating_weight, args.h2h_weight, args.style_weight)
    print(f"Mean Squared Error: {mse}")