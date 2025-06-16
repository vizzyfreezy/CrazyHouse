# main_data_collector.py

import requests
import json
import time
from datetime import datetime, timezone
import math
import os

USERNAME = 'Bayormiller_CNO'
START_TIMESTAMP = int(datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
CACHE_DIR = 'tournament_cache'
DATA_OUTPUT_DIR = 'data_output' # New directory for saving processed data

# Ensure directories exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(DATA_OUTPUT_DIR):
    os.makedirs(DATA_OUTPUT_DIR)

# Global dicts to store processed data
medal_table = {}
player_overall_stats = {}
berserk_table = {}

# ... (All your existing fetch and processing functions:
# load_from_cache, save_to_cache,
# fetch_created_tournaments, fetch_tournament_details,
# process_tournament, process_medals, update_player_overall_stats,
# count_berserk_wins_from_sheet, process_berserk - These remain largely the same) ...

# --- Function to save processed data ---
def save_processed_data():
    """Saves all the collected and processed data tables to JSON files."""
    print("\nSaving processed data...")
    try:
        with open(os.path.join(DATA_OUTPUT_DIR, 'medal_table.json'), 'w') as f:
            json.dump(medal_table, f, indent=4)
        
        with open(os.path.join(DATA_OUTPUT_DIR, 'player_overall_stats.json'), 'w') as f:
            json.dump(player_overall_stats, f, indent=4)

        with open(os.path.join(DATA_OUTPUT_DIR, 'berserk_table.json'), 'w') as f:
            json.dump(berserk_table, f, indent=4)
        
        print(f"Data saved successfully to '{DATA_OUTPUT_DIR}' directory.")
    except IOError as e:
        print(f"Error saving data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving data: {e}")


# --- Main Function (Modified) ---
def main():
    global player_overall_stats, medal_table, berserk_table # Ensure they are accessible
    player_overall_stats = {} # Initialize/clear for each run
    medal_table = {}
    berserk_table = {}
    
    processed_tournaments_count = 0
    eligible_tournaments_found = 0

    print(f"Fetching tournaments created by {USERNAME} starting from {datetime.fromtimestamp(START_TIMESTAMP/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}...")

    for tournament_summary in fetch_created_tournaments():
        # ... (your existing tournament fetching and filtering logic) ...
        starts_at_val = tournament_summary.get('startsAt')
        current_tournament_ts = 0
        if isinstance(starts_at_val, (int, float)):
            current_tournament_ts = int(starts_at_val)
        elif isinstance(starts_at_val, str):
            try:
                dt_obj = datetime.fromisoformat(starts_at_val.replace('Z', '+00:00'))
                if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                current_tournament_ts = int(dt_obj.timestamp() * 1000)
            except ValueError: continue
        else: continue

        if current_tournament_ts >= START_TIMESTAMP:
            eligible_tournaments_found += 1
            tid = tournament_summary.get('id')
            if not tid: continue
            
            details = fetch_tournament_details(tid)
            if details:
                process_tournament(details)
                processed_tournaments_count += 1
    
    print(f"\nFound {eligible_tournaments_found} tournaments since the start date.")
    print(f"Successfully processed {processed_tournaments_count} tournaments meeting all criteria.")
    
    if processed_tournaments_count > 0:
        save_processed_data() # Save the data instead of printing/generating images
    else:
        print("\nNo tournaments processed. No data saved.")

if __name__ == '__main__':
    def load_from_cache(tournament_id):
        cache_file = os.path.join(CACHE_DIR, f"{tournament_id}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                is_finished_in_cache = cached_data.get('finished', False) or \
                                    cached_data.get('status') == 'finished' or \
                                    cached_data.get('isFinished', False)
                if is_finished_in_cache:
                    return cached_data
            except Exception: pass # Simplified error handling for brevity
        return None

    def save_to_cache(tournament_id, data):
        if not os.path.exists(CACHE_DIR):
            try: os.makedirs(CACHE_DIR)
            except OSError: return
        cache_file = os.path.join(CACHE_DIR, f"{tournament_id}.json")
        try:
            with open(cache_file, 'w') as f: json.dump(data, f)
        except Exception: pass # Simplified

    def fetch_created_tournaments():
        url = f'https://lichess.org/api/user/{USERNAME}/tournament/created'
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try: yield json.loads(line.decode())
                    except json.JSONDecodeError: pass
        except requests.exceptions.RequestException: return

    def fetch_tournament_details(tournament_id):
        cached_data = load_from_cache(tournament_id)
        if cached_data: return cached_data
        base_url = f"https://lichess.org/api/tournament/{tournament_id}"
        time.sleep(0.5) # Adjusted sleep
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            tournament_data = response.json()
        except Exception: return None

        nbPlayers = tournament_data.get('nbPlayers', 0)
        standing = tournament_data.get('standing', {})
        if nbPlayers == 0 or not standing.get('players'):
            if tournament_data.get('finished', False) or tournament_data.get('status') == 'finished' or tournament_data.get('isFinished', False):
                save_to_cache(tournament_id, tournament_data)
            return tournament_data
        all_players = list(standing.get('players', []))
        players_per_page = 10
        total_pages = math.ceil(nbPlayers / players_per_page)
        current_api_page = standing.get('page', 1)
        while current_api_page < total_pages:
            next_page_to_fetch = current_api_page + 1
            next_page_url = f"{base_url}?page={next_page_to_fetch}"
            time.sleep(0.5) # Adjusted sleep
            try:
                next_page_response = requests.get(next_page_url)
                next_page_response.raise_for_status()
                next_page_data = next_page_response.json()
            except Exception: break
            next_page_players_list = next_page_data.get('standing', {}).get('players', [])
            if not next_page_players_list: break
            all_players.extend(next_page_players_list)
            current_api_page += 1
        seen_players = set()
        unique_players = [p for p in all_players if p.get('name') and p['name'] not in seen_players and not seen_players.add(p['name'])]
        tournament_data['players'] = unique_players
        if tournament_data.get('finished', False) or tournament_data.get('status') == 'finished' or tournament_data.get('isFinished', False):
            save_to_cache(tournament_id, tournament_data)
        return tournament_data

    def process_tournament(data):
        stats = data.get('stats')
        if not stats or stats.get('games', 0) < 15: return
        process_medals(data)
        update_player_overall_stats(data)
        process_berserk(data)

    def process_medals(data):
        for medal in data.get('podium', []):
            name = medal.get('name')
            if not name: continue
            rank = medal.get('rank')
            if rank is None: continue
            medal_table.setdefault(name, {'gold': 0, 'silver': 0, 'bronze': 0})
            if rank == 1: medal_table[name]['gold'] += 1
            elif rank == 2: medal_table[name]['silver'] += 1
            elif rank == 3: medal_table[name]['bronze'] += 1

    def update_player_overall_stats(data):
        tournament_players_processed_for_participation = set() 
        min_games_for_active_tournament_credit = 5 
        for player_data in data.get('players', []): 
            name = player_data.get('name')
            if not name: continue
            score = player_data.get('score', 0)
            sheet_scores = player_data.get('sheet', {}).get('scores', "")
            games_in_this_tournament = len(sheet_scores) if isinstance(sheet_scores, str) else 0
            current_player_stats = player_overall_stats.setdefault(name, {'total_score': 0, 'total_games': 0, 'tournaments_played_min_5_games': 0})
            current_player_stats['total_score'] += score
            current_player_stats['total_games'] += games_in_this_tournament
            if games_in_this_tournament >= min_games_for_active_tournament_credit:
                if name not in tournament_players_processed_for_participation:
                    current_player_stats['tournaments_played_min_5_games'] += 1
                    tournament_players_processed_for_participation.add(name)

    def count_berserk_wins_from_sheet(scores_string):
        if isinstance(scores_string, str): return scores_string.count('3') + scores_string.count('5')
        return 0

    def process_berserk(data):
        player_lookup = {p.get('name'): p for p in data.get('players', []) if p.get('name')}
        for p_podium in data.get('podium', []):
            name = p_podium.get('name')
            if not name: continue
            nb_stats = p_podium.get('nb', {}) 
            berserks_played_this_tourney = nb_stats.get('berserk', 0)
            total_games_for_podium_player_this_tourney = nb_stats.get('game', 0) 
            if berserks_played_this_tourney == 0: continue
            full_player_data = player_lookup.get(name)
            if full_player_data:
                sheet = full_player_data.get('sheet', {})
                scores_string = sheet.get('scores', "") 
                berserk_wins_counted = count_berserk_wins_from_sheet(scores_string)
                entry = berserk_table.setdefault(name, {"berserk_games": 0, "berserk_wins": 0, "total_games_in_podium_summary": 0})
                entry["berserk_games"] += berserks_played_this_tourney
                entry["berserk_wins"] += berserk_wins_counted
                entry["total_games_in_podium_summary"] += total_games_for_podium_player_this_tourney

    main()