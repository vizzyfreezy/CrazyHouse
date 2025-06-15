import requests
import json
import time
from datetime import datetime, timezone
import math
import os

USERNAME = 'Bayormiller_CNO'
START_TIMESTAMP = int(datetime(2025, 3, 22, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
CACHE_DIR = 'tournament_cache'

medal_table = {}
# Key name changed for clarity: 'tournaments_played_min_5_games'
player_overall_stats = {} # Stores {'name': {'total_score': X, 'total_games': Y, 'tournaments_played_min_5_games': Z}}
berserk_table = {}

# --- Cache Utility Functions (No changes here) ---
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
        except json.JSONDecodeError:
            print(f"Warning: Corrupted cache file for {tournament_id}, removing.")
            try:
                os.remove(cache_file)
            except OSError as e:
                print(f"Warning: Could not remove corrupted cache file {cache_file}: {e}")
        except Exception as e:
            print(f"Warning: Error loading cache for {tournament_id}: {e}, will re-fetch.")
    return None

def save_to_cache(tournament_id, data):
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
        except OSError as e:
            print(f"Error: Could not create cache directory {CACHE_DIR}: {e}. Cannot save cache.")
            return
    cache_file = os.path.join(CACHE_DIR, f"{tournament_id}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error: Could not save cache for {tournament_id}: {e}")

def fetch_created_tournaments():
    url = f'https://lichess.org/api/user/{USERNAME}/tournament/created'
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                try:
                    yield json.loads(line.decode())
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line from created tournaments stream: {e} - Line: {line[:100]}...")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching created tournaments list: {e}")
        return

def fetch_tournament_details(tournament_id):
    cached_data = load_from_cache(tournament_id)
    if cached_data:
        return cached_data

    base_url = f"https://lichess.org/api/tournament/{tournament_id}"
    time.sleep(1)
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        tournament_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tournament details for {tournament_id} (initial page): {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for tournament {tournament_id} (initial page): {e}")
        return None

    nbPlayers = tournament_data.get('nbPlayers', 0)
    standing = tournament_data.get('standing', {})

    if nbPlayers == 0 or not standing.get('players'):
        is_finished_api = tournament_data.get('finished', False) or \
                          tournament_data.get('status') == 'finished' or \
                          tournament_data.get('isFinished', False)
        if is_finished_api:
            save_to_cache(tournament_id, tournament_data)
        return tournament_data

    all_players = list(standing.get('players', []))
    players_per_page = 10
    total_pages = math.ceil(nbPlayers / players_per_page)
    current_api_page = standing.get('page', 1)

    while current_api_page < total_pages:
        next_page_to_fetch = current_api_page + 1
        next_page_url = f"{base_url}?page={next_page_to_fetch}"
        time.sleep(1)
        try:
            next_page_response = requests.get(next_page_url)
            next_page_response.raise_for_status()
            next_page_data = next_page_response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching tournament details for {tournament_id} (page {next_page_to_fetch}): {e}")
            break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for tournament {tournament_id} (page {next_page_to_fetch}): {e}")
            break

        next_page_players_list = next_page_data.get('standing', {}).get('players', [])
        if not next_page_players_list:
            break
        all_players.extend(next_page_players_list)
        current_api_page += 1

    seen_players = set()
    unique_players = []
    for p in all_players:
        player_name = p.get('name')
        if player_name and player_name not in seen_players:
            seen_players.add(player_name)
            unique_players.append(p)
        elif not player_name:
            print(f"Warning: Player object without name in tournament {tournament_id}: {p}")
    
    tournament_data['players'] = unique_players

    is_finished_api = tournament_data.get('finished', False) or \
                      tournament_data.get('status') == 'finished' or \
                      tournament_data.get('isFinished', False)
    if is_finished_api:
        save_to_cache(tournament_id, tournament_data)
    return tournament_data

# --- Processing Functions ---
def process_tournament(data):
    stats = data.get('stats')
    if not stats or stats.get('games', 0) < 15:
        return

    process_medals(data)
    update_player_overall_stats(data) # This function is now modified
    process_berserk(data)

def process_medals(data): # Unchanged
    for medal in data.get('podium', []):
        name = medal.get('name')
        if not name: continue
        rank = medal.get('rank')
        if rank is None: continue
            
        medal_table.setdefault(name, {'gold': 0, 'silver': 0, 'bronze': 0})
        if rank == 1: medal_table[name]['gold'] += 1
        elif rank == 2: medal_table[name]['silver'] += 1
        elif rank == 3: medal_table[name]['bronze'] += 1

def update_player_overall_stats(data): # MODIFIED
    # This function is called once per processed tournament.
    tournament_players_processed_for_participation = set() 

    min_games_for_active_tournament_credit = 5 # Define the threshold here

    for player_data in data.get('players', []): 
        name = player_data.get('name')
        if not name:
            print(f"Warning: Player entry without name during stats update for {data.get('id', 'Unknown ID')}")
            continue
        
        score = player_data.get('score', 0)
        sheet_scores = player_data.get('sheet', {}).get('scores', "")
        # Calculate games played in THIS specific tournament
        games_in_this_tournament = len(sheet_scores) if isinstance(sheet_scores, str) else 0

        current_player_stats = player_overall_stats.setdefault(
            name, 
            {'total_score': 0, 'total_games': 0, 'tournaments_played_min_5_games': 0} # Key name updated
        )
        
        current_player_stats['total_score'] += score
        current_player_stats['total_games'] += games_in_this_tournament # Accumulate total games across all tournaments
        
        # Only increment 'tournaments_played_min_5_games' if the player played enough games IN THIS TOURNAMENT
        if games_in_this_tournament >= min_games_for_active_tournament_credit:
            # Ensure we only credit participation once per player for this tournament, even if they appeared multiple times in list
            if name not in tournament_players_processed_for_participation:
                current_player_stats['tournaments_played_min_5_games'] += 1
                tournament_players_processed_for_participation.add(name)

def count_berserk_wins_from_sheet(scores_string): # Unchanged
    if isinstance(scores_string, str):
        return scores_string.count('3') + scores_string.count('5')
    return 0

def process_berserk(data): # Unchanged
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
        else:
            print(f"Warning: Podium player {name} not found in main player list for detailed berserk stats in tournament {data.get('id', 'Unknown ID')}.")


# --- Printing Functions ---
def print_medal_table(): # Unchanged
    print("\n=== Medal Table ===")
    leaderboard = [(name, medals) for name, medals in medal_table.items() if sum(medals.values()) > 0]
    leaderboard.sort(key=lambda x: (-x[1]['gold'], -x[1]['silver'], -x[1]['bronze'], x[0].lower()))
    for i, (player, medals) in enumerate(leaderboard, 1):
        print(f"{i}. {player:<20} Gold: {medals['gold']}, Silver: {medals['silver']}, Bronze: {medals['bronze']}")

def print_berserk_table(top_n=20, min_berserk_games_overall=5): # Unchanged
    print(f"\n=== Berserk Heroes (Top {top_n}, ≥{min_berserk_games_overall} total berserk games from podium appearances) ===")
    valid_stats = []
    for name, rec in berserk_table.items():
        if rec.get("berserk_games", 0) >= min_berserk_games_overall:
            win_rate = (rec["berserk_wins"] / rec["berserk_games"]) if rec["berserk_games"] > 0 else 0
            berserk_freq = (rec["berserk_games"] / rec["total_games_in_podium_summary"]) if rec.get("total_games_in_podium_summary", 0) > 0 else 0 
            valid_stats.append({
                "name": name, "win_rate": win_rate, "berserk_games": rec["berserk_games"],
                "berserk_freq": berserk_freq, "total_games_podium": rec.get("total_games_in_podium_summary", 0)
            })
    print(f"\n--- Sorted by Berserk Win Rate (Top {top_n}) ---")
    stats_by_win_rate = sorted(valid_stats, key=lambda x: (-x["win_rate"], -x["berserk_games"], x["name"].lower()))
    for i, s in enumerate(stats_by_win_rate[:top_n], 1):
        print(f"{i}. {s['name']:<20} Win rate: {s['win_rate']:.1%} (berserk games: {s['berserk_games']})")
    print(f"\n--- Sorted by Berserk Frequency (in podium games) (Top {top_n}) ---")
    stats_by_freq = sorted(valid_stats, key=lambda x: (-x["berserk_freq"], -x["berserk_games"], x["name"].lower()))
    for i, s in enumerate(stats_by_freq[:top_n], 1):
        print(f"{i}. {s['name']:<20} Berserk Freq: {s['berserk_freq']:.1%} ({s['berserk_games']} of {s['total_games_podium']} podium summary games)")

def print_points_table(top_n=15): # Unchanged
    print(f"\n=== Points Table (Top {top_n}) ===")
    leaderboard_data = [(name, stats['total_score']) for name, stats in player_overall_stats.items()]
    leaderboard_data.sort(key=lambda x: -x[1])
    for i, (player, pts) in enumerate(leaderboard_data[:top_n], 1):
        print(f"{i}. {player:<20} Points: {pts}")

def print_points_per_game_table(top_n=10, min_total_games=10): # Unchanged
    print(f"\n=== Points Per Game (PPG) Table (Top {top_n}, ≥{min_total_games} total games) ===")
    ppg_leaderboard = []
    for name, stats in player_overall_stats.items():
        total_games = stats.get('total_games', 0)
        total_score = stats.get('total_score', 0)
        if total_games >= min_total_games:
            ppg = (total_score / total_games) if total_games > 0 else 0
            ppg_leaderboard.append({
                "name": name, "ppg": ppg, "total_score": total_score, "total_games": total_games
            })
    ppg_leaderboard.sort(key=lambda x: (-x["ppg"], -x["total_games"], x["name"].lower()))
    for i, entry in enumerate(ppg_leaderboard[:top_n], 1):
        print(f"{i}. {entry['name']:<20} PPG: {entry['ppg']:.2f} (Points: {entry['total_score']}, Games: {entry['total_games']})")

# --- MODIFIED Printing Function (Name and logic updated) ---
def print_most_active_tournaments_table(top_n=15, min_games_per_tournament_for_credit=5):
    print(f"\n=== Most Active Tournaments (Played ≥{min_games_per_tournament_for_credit} games per tournament) (Top {top_n}) ===")
    
    participation_data = []
    for name, stats in player_overall_stats.items():
        # Use the new key that reflects the condition
        tournaments_count = stats.get('tournaments_played_min_5_games', 0) 
        if tournaments_count > 0:
            participation_data.append((name, tournaments_count))
            
    participation_data.sort(key=lambda x: (-x[1], x[0].lower()))
    
    for i, (player, count) in enumerate(participation_data[:top_n], 1):
        print(f"{i}. {player:<20} Active Tournaments: {count}")

# --- Main Function (Call to the print function updated) ---
def main():
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
            print(f"Created cache directory: {CACHE_DIR}")
        except OSError as e:
            print(f"Warning: Could not create cache directory {CACHE_DIR}: {e}. Caching may fail.")

    processed_tournaments_count = 0
    eligible_tournaments_found = 0
    print(f"Fetching tournaments created by {USERNAME} starting from {datetime.fromtimestamp(START_TIMESTAMP/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}...")

    for tournament_summary in fetch_created_tournaments():
        starts_at_val = tournament_summary.get('startsAt')
        current_tournament_ts = 0
        # ... (timestamp parsing logic - unchanged) ...
        if isinstance(starts_at_val, (int, float)):
            current_tournament_ts = int(starts_at_val)
        elif isinstance(starts_at_val, str):
            try:
                dt_obj = datetime.fromisoformat(starts_at_val.replace('Z', '+00:00'))
                if dt_obj.tzinfo is None: dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                current_tournament_ts = int(dt_obj.timestamp() * 1000)
            except ValueError:
                print(f"Warning: Could not parse 'startsAt' string: '{starts_at_val}' for tournament {tournament_summary.get('id')}")
                continue
        else:
            print(f"Warning: Unknown type for 'startsAt': {type(starts_at_val)} for tournament {tournament_summary.get('id')}")
            continue

        if current_tournament_ts >= START_TIMESTAMP:
            eligible_tournaments_found += 1
            tid = tournament_summary.get('id')
            if not tid:
                print(f"Warning: Tournament summary missing 'id': {tournament_summary}")
                continue
            
            print(f"Processing tournament {tid} (Starts: {datetime.fromtimestamp(current_tournament_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')})")
            details = fetch_tournament_details(tid)
            if details:
                process_tournament(details)
                processed_tournaments_count += 1
            else:
                print(f"Info: Could not retrieve or process details for tournament {tid}.")
    
    print(f"\nFound {eligible_tournaments_found} tournaments since the start date.")
    print(f"Successfully processed {processed_tournaments_count} tournaments meeting all criteria.")
    
    if processed_tournaments_count > 0:
        print_medal_table()
        print_points_table()
        print_berserk_table(top_n=10, min_berserk_games_overall=5)
        print_points_per_game_table(top_n=10, min_total_games=20)
        # Updated call with the min_games_per_tournament_for_credit parameter, matching the default in the function
        print_most_active_tournaments_table(top_n=15, min_games_per_tournament_for_credit=5) 
    else:
        print("\nNo tournaments processed. Tables will be empty.")

if __name__ == '__main__':
    # For local testing, ensure you have the CACHE_DIR created or permissions to create it.
    # And that USERNAME and START_TIMESTAMP are set appropriately.
    main()