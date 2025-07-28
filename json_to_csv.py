
import json
import os
import csv

SOURCE_DIR = "detailed_games"
OUTPUT_FILE = "games.csv"

def main():
    """
    Reads all JSON files from the detailed_games directory,
    flattens the data, and writes it to a single CSV file.
    """
    all_games = []
    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(SOURCE_DIR, filename)
            with open(filepath, "r") as f:
                try:
                    games_data = json.load(f)
                    # Each file contains a list of games
                    for game in games_data:
                        all_games.append(game)
                except json.JSONDecodeError:
                    print(f"Could not decode JSON from {filename}")

    if not all_games:
        print("No game data found to process.")
        return

    # Define the headers for the CSV file
    headers = [
        "id",
        "date",
        "time",
        "time_control",
        "winner_name",
        "winner_rating",
        "loser_name",
        "loser_rating",
        "moves",
        "status",
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_games)

    print(f"Successfully converted {len(all_games)} games to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
