import requests
import json
import os
import time
import chess.pgn
import io
from datetime import datetime, timezone
import argparse

# --- CONFIGURATION (Remains the same) ---
SOURCE_DIR = "tournament_cache"
DEST_DIR = "detailed_games"
API_ENDPOINT = "https://lichess.org/api/tournament/{}/games"
RATE_LIMIT_DELAY_SECONDS = 0
START_DATE = "2025-02-15"
END_DATE = "2026-07-29"

def parse_iso_zulu(date_str):
    """wrapper function for parsing datetime objects in which my have fractional seconds or not """
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {date_str}")

def parse_pgn_stream(pgn_text):
    """
    Parses a string containing multiple PGN games and now includes
    the color of the winning and losing players.
    """
    pgn_file_stream = io.StringIO(pgn_text)
    games_list = []

    while True:
        game = chess.pgn.read_game(pgn_file_stream)
        if game is None:
            break

        headers = game.headers
        result = headers.get("Result", "*")
        
        # Get the names of the White and Black players
        white_player = headers.get("White")
        black_player = headers.get("Black")
        white_rating = int(headers.get("WhiteElo"))
        black_rating = int(headers.get("BlackElo"))

        # --- THE KEY CHANGE IS HERE ---
        # Determine winner/loser and also capture their color for that game.
        if result == "1-0":
            winner_name = white_player
            winner_rating = white_rating
            winner_color = "White" # Winner played White
            loser_name = black_player
            loser_rating = black_rating
            loser_color = "Black" # Loser played Black
        elif result == "0-1":
            winner_name = black_player
            winner_rating = black_rating
            winner_color = "Black" # Winner played Black
            loser_name = white_player
            loser_rating = white_rating
            loser_color = "White" # Loser played White
        else:
            # Skip draws or unfinished games
            continue

        # Manual move counting logic (remains the same)
        try:
            moves_section = str(game).split("\n\n", 1)[1]
        except IndexError:
            moves_section = ""
        tokens = moves_section.split()
        move_tokens = [
            token for token in tokens
            if not token.endswith(".") and token not in ["1-0", "0-1", "1/2-1/2", "*"]
        ]
        move_count = (len(move_tokens) + 1) // 2

        # --- Add the new color fields to the output ---
        games_list.append(
            {
                "id": headers.get("Site", "").split("/")[-1],
                "date": headers.get("UTCDate", ""),
                "time": headers.get("UTCTime", ""),
                "time_control": headers.get("TimeControl", ""),
                "winner_name": winner_name,
                "winner_rating": winner_rating,
                "winner_color": winner_color, # New field
                "loser_name": loser_name,
                "loser_rating": loser_rating,
                "loser_color": loser_color,   # New field
                "moves": move_count,
                "status": headers.get("Termination", "Unknown"),
            }
        )

    return games_list
# The rest of the file remains exactly the same
def fetch_and_parse_tournament(tournament_id):
    url = API_ENDPOINT.format(tournament_id)
    print(f"Fetching PGN from {url}...")
    try:
        response = requests.get(url, headers={"Accept": "application/x-chess-pgn"})
        response.raise_for_status()
        return parse_pgn_stream(response.text)
    except requests.exceptions.RequestException as e:
        print(
            f"  ERROR: Could not fetch data for tournament {tournament_id}. Reason: {e}"
        )
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch and parse tournament games.")
    parser.add_argument("tournament_id", nargs='?', default=None, help="The ID of the tournament to fetch.")
    args = parser.parse_args()

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    if args.tournament_id:
        print(f"--- Processing single tournament: {args.tournament_id} ---")
        dest_filepath = os.path.join(DEST_DIR, f"{args.tournament_id}.json")
        if os.path.exists(dest_filepath):
            print(f"Skipping {args.tournament_id}: Detailed game file already exists.")
            return

        games = fetch_and_parse_tournament(args.tournament_id)
        if games is not None:
            with open(dest_filepath, "w") as f:
                json.dump(games, f, indent=2)
            print(f"  SUCCESS: Saved {len(games)} games to {dest_filepath}")
        return

    start_dt = datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(END_DATE + "T23:59:59").replace(tzinfo=timezone.utc)
    print(f"--- Processing tournaments between {START_DATE} and {END_DATE} ---")
    summary_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".json")]
    for filename in summary_files:
        tournament_id = os.path.splitext(filename)[0]
        source_filepath = os.path.join(SOURCE_DIR, filename)
        try:
            with open(source_filepath, "r") as f:
                summary_data = json.load(f)
            date_str = summary_data.get("startsAt")
            if not date_str:
                continue
            tournament_dt =parse_iso_zulu(date_str)
            if not (start_dt <= tournament_dt <= end_dt):
                continue
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Skipping {tournament_id}: Error reading summary file. Reason: {e}")
            continue
        dest_filepath = os.path.join(DEST_DIR, f"{tournament_id}.json")
        if os.path.exists(dest_filepath):
            print(f"Skipping {tournament_id}: Detailed game file already exists.")
            continue
        print(f"Processing {tournament_id} (Date: {tournament_dt.date()})")
        games = fetch_and_parse_tournament(tournament_id)
        if games is not None:
            with open(dest_filepath, "w") as f:
                json.dump(games, f, indent=2)
            print(f"  SUCCESS: Saved {len(games)} games to {dest_filepath}")
        print(f"Waiting for {RATE_LIMIT_DELAY_SECONDS} seconds...")
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
    print("\nDate-filtered processing complete!")


if __name__ == "__main__":
    main()