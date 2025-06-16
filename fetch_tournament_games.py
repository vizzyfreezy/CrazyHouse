import requests
import json
import os
import time
import chess.pgn
import io
from datetime import datetime, timezone

# --- CONFIGURATION (Remains the same) ---
SOURCE_DIR = "tournament_cache"
DEST_DIR = "detailed_games"
API_ENDPOINT = "https://lichess.org/api/tournament/{}/games"
RATE_LIMIT_DELAY_SECONDS = 0
START_DATE = "2025-06-01"
END_DATE = "2025-06-15"


def parse_pgn_stream(pgn_text):
    """
    Parses a string containing multiple PGN games.
    THIS FUNCTION CONTAINS THE DEFINITIVE FIX.
    """
    pgn_file_stream = io.StringIO(pgn_text)
    games_list = []

    while True:
        # We still use read_game() because it's excellent at isolating one game
        # block and parsing its headers, which is a complex task to do manually.
        game = chess.pgn.read_game(pgn_file_stream)
        if game is None:
            break

        headers = game.headers
        result = headers.get('Result', '*')

        if result == "1-0":
            winner_name = headers.get('White')
            winner_rating = int(headers.get('WhiteElo'))
            loser_name = headers.get('Black')
            loser_rating = int(headers.get('BlackElo'))
        elif result == "0-1":
            winner_name = headers.get('Black')
            winner_rating = int(headers.get('BlackElo'))
            loser_name = headers.get('White')
            loser_rating = int(headers.get('WhiteElo'))
        else:
            continue

        # ======================================================================
        # --- THE FIX: MANUAL MOVE COUNTING ---
        # We convert the game object back to its raw PGN string, which is easy to work with.
        game_string = str(game)
        
        # The moves always come after a double newline that separates them from the headers.
        # We find that spot and take everything after it.
        try:
            moves_section = game_string.split('\n\n', 1)[1]
        except IndexError:
            # This happens if a game has headers but no moves.
            moves_section = ""

        # We split the move text into tokens by spaces.
        # e.g., "1. d4 d5 0-1" becomes ["1.", "d4", "d5", "0-1"]
        tokens = moves_section.split()
        
        # We now filter this list to ONLY include actual move tokens.
        # We exclude move numbers (like "1.") and results.
        move_tokens = [
            token for token in tokens 
            if not token.endswith('.') and token not in ["1-0", "0-1", "1/2-1/2", "*"]
        ]
        
        # The number of tokens is the number of half-moves (plies).
        # We can now reliably calculate the full move count.
        num_plies = len(move_tokens)
        move_count = (num_plies + 1) // 2
        # ======================================================================

        games_list.append({
            "id": headers.get('Site', '').split('/')[-1],
            "winner_name": winner_name, "winner_rating": winner_rating,
            "loser_name": loser_name, "loser_rating": loser_rating,
            "moves": move_count,
            "status": headers.get('Termination', 'Unknown')
        })

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
        print(f"  ERROR: Could not fetch data for tournament {tournament_id}. Reason: {e}")
        return None

def main():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    start_dt = datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(END_DATE + "T23:59:59").replace(tzinfo=timezone.utc)
    print(f"--- Processing tournaments between {START_DATE} and {END_DATE} ---")
    summary_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]
    for filename in summary_files:
        tournament_id = os.path.splitext(filename)[0]
        source_filepath = os.path.join(SOURCE_DIR, filename)
        try:
            with open(source_filepath, 'r') as f:
                summary_data = json.load(f)
            date_str = summary_data.get('startsAt')
            if not date_str:
                continue
            tournament_dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
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
            with open(dest_filepath, 'w') as f:
                json.dump(games, f, indent=2)
            print(f"  SUCCESS: Saved {len(games)} games to {dest_filepath}")
        print(f"Waiting for {RATE_LIMIT_DELAY_SECONDS} seconds...")
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
    print("\nDate-filtered processing complete!")

if __name__ == "__main__":
    main()