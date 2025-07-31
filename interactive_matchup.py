import pandas as pd
import numpy as np
import random
import os
import subprocess

# Import functions directly from simulate_teambattle.py
# This assumes simulate_teambattle.py is in the same directory
from simulate_teambattle import (
    calculate_h2h_advantage,
    calculate_style_advantage,
    simulate_game_advanced,
    build_h2h_wins_dict_from_games, # Renamed from build_h2h_data_from_games
    WEIGHT_RATING, WEIGHT_H2H, WEIGHT_STYLE, DRAW_PROBABILITY
)

# ==============================================================================
# INTERACTIVE SCRIPT LOGIC
# ==============================================================================

if __name__ == "__main__":
    try:
        player_data_df = pd.read_csv('player_features.csv', index_col='player_name')
        games_df = pd.read_csv('all_games.csv')
    except FileNotFoundError:
        print("\n!!! ERROR: CSV files not found. Please ensure 'player_features.csv' and 'all_games.csv' are in the current directory.")
        exit()

    player_data_dict = player_data_df.to_dict('index')
    h2h_wins_dict = build_h2h_wins_dict_from_games(games_df)

    # Prepare player list for fzf with ratings and sorting
    player_ratings_list = []
    for player_name in player_data_dict.keys():
        rating = player_data_dict[player_name]['Current Rating']
        player_ratings_list.append((rating, player_name))

    # Sort by rating in descending order
    player_ratings_list.sort(key=lambda x: x[0], reverse=True)

    # Format for fzf display
    fzf_display_list = [f"{name} ({rating:.0f})" for rating, name in player_ratings_list]

    print("--- Interactive Player Matchup Simulation ---")
    print("Select two players to simulate a game between them.")
    print("Press Ctrl+C to exit.")

    while True:
        try:
            # Player 1 selection
            print("\nSelect Player 1:")
            player1_selection = subprocess.run(
                ['fzf', '--prompt', 'Player 1 > ', '--height', '40%', '--layout', 'reverse', '--border'],
                input="\n".join(fzf_display_list),
                capture_output=True, text=True, check=True
            ).stdout.strip()

            if not player1_selection:
                print("No player selected. Exiting.")
                break
            
            p1_name = player1_selection.split(' (')[0] # Extract name before ' ('
            p1_stats = player_data_dict[p1_name]

            # Player 2 selection
            print("\nSelect Player 2:")
            # Filter out already selected player and prepare for fzf
            fzf_display_list_p2 = [item for item in fzf_display_list if not item.startswith(p1_name + ' (')]
            player2_selection = subprocess.run(
                ['fzf', '--prompt', 'Player 2 > ', '--height', '40%', '--layout', 'reverse', '--border'],
                input="\n".join(fzf_display_list_p2),
                capture_output=True, text=True, check=True
            ).stdout.strip()

            if not player2_selection:
                print("No player selected. Exiting.")
                break

            p2_name = player2_selection.split(' (')[0]
            p2_stats = player_data_dict[p2_name]

            print(f"\n--- Matchup: {p1_name} vs {p2_name} ---")
            print(f"  {p1_name} Rating: {p1_stats['Current Rating']:.0f}")
            print(f"  {p2_name} Rating: {p2_stats['Current Rating']:.0f}")

            p1_h2h_wins = h2h_wins_dict.get((p1_name, p2_name), 0)
            p2_h2h_wins = h2h_wins_dict.get((p2_name, p1_name), 0)
            print(f"  H2H Record: {p1_name} {p1_h2h_wins} - {p2_h2h_wins} {p2_name}")
            print(f"  (H2H advantage only applies if total games >= 5)")

            while True:
                choice = input("\nPress Enter to simulate 10 games (or type 'n' for new players, 'q' to quit): ").strip().lower()
                if choice == 'n':
                    break
                elif choice == 'q':
                    exit()
                elif choice != '': # If not empty, and not 'n' or 'q', then it's an invalid input
                    print("Invalid input. Please press Enter, or type 'n' or 'q'.")
                    continue

                p1_wins = 0
                p2_wins = 0
                draws = 0
                num_simulations = 100

                print(f"\nSimulating {num_simulations} games...")
                for _ in range(num_simulations):
                    p1_score, p2_score = simulate_game_advanced(p1_name, p2_name, p1_stats, p2_stats, h2h_wins_dict)
                    if p1_score == 1.0:
                        p1_wins += 1
                    elif p2_score == 1.0:
                        p2_wins += 1
                    else:
                        draws += 1
                
                print(f"\n--- Simulation Results ({num_simulations} games) ---")
                print(f"  {p1_name} wins: {p1_wins} ({p1_wins / num_simulations:.1%})")
                print(f"  {p2_name} wins: {p2_wins} ({p2_wins / num_simulations:.1%})")
                print(f"  Draws: {draws} ({draws / num_simulations:.1%})")

        except subprocess.CalledProcessError:
            print("\nFZF command failed. Make sure fzf is installed and in your PATH.")
            print("You can install it via: apt install fzf (on Termux/Android)")
            break
        except KeyError as e:
            print(f"Error: Player not found in data: {e}. Please check player names.")
            break
        except KeyboardInterrupt:
            print("\nExiting interactive simulation.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break