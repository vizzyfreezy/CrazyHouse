# html_report_generator.py
import json
import os
from datetime import datetime, timezone
from jinja2 import Environment, FileSystemLoader
from zoneinfo import ZoneInfo
DATA_DIR = 'data_output'
TEMPLATE_DIR = 'templates'
HTML_OUTPUT_FILE = 'Cno_Arena_Report.html'
USERNAME_FOR_REPORT = 'Bayormiller_CNO' # Or get from config/env

def load_data(filename):
    """Loads JSON data from the specified file."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found - {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {filepath}")
        return None
    except IOError:
        print(f"Warning: Could not read file {filepath}")
        return None

def prepare_medal_data(medal_table_raw):
    if not medal_table_raw: return []
    leaderboard = []
    # Sort raw data
    temp_leaderboard = sorted(
        medal_table_raw.items(), 
        key=lambda x: (-x[1]['gold'], -x[1]['silver'], -x[1]['bronze'], x[0].lower())
    )
    for player, medals in temp_leaderboard:
        if sum(medals.values()) > 0:
            leaderboard.append({"player": player, **medals})
    return leaderboard

def prepare_points_data(player_stats_raw, top_n=10):
    if not player_stats_raw: return []
    leaderboard = sorted(
        player_stats_raw.items(),
        key=lambda x: -x[1].get('total_score', 0)
    )
    return [{"player": name, "points": stats.get('total_score',0)} for name, stats in leaderboard[:top_n]]

def prepare_berserk_data(berserk_table_raw, top_n=10, min_games=5):
    if not berserk_table_raw: return [], []
    valid_stats = []
    for name, rec in berserk_table_raw.items():
        if rec.get("berserk_games", 0) >= min_games:
            win_rate = (rec["berserk_wins"] / rec["berserk_games"]) if rec["berserk_games"] > 0 else 0
            berserk_freq = (rec["berserk_games"] / rec["total_games_in_podium_summary"]) if rec.get("total_games_in_podium_summary", 0) > 0 else 0 
            valid_stats.append({
                "name": name, "win_rate": win_rate, "berserk_games": rec["berserk_games"],
                "berserk_freq": berserk_freq, "total_games_podium": rec.get("total_games_in_podium_summary", 0)
            })
    
    stats_by_win_rate = sorted(valid_stats, key=lambda x: (-x["win_rate"], -x["berserk_games"], x["name"].lower()))
    stats_by_freq = sorted(valid_stats, key=lambda x: (-x["berserk_freq"], -x["berserk_games"], x["name"].lower()))
    
    return stats_by_win_rate[:top_n], stats_by_freq[:top_n]

def prepare_ppg_data(player_stats_raw, top_n=10, min_total_games=20):
    if not player_stats_raw: return []
    ppg_leaderboard_prep = []
    for name, stats in player_stats_raw.items():
        total_games = stats.get('total_games', 0)
        total_score = stats.get('total_score', 0)
        if total_games >= min_total_games:
            ppg = (total_score / total_games) if total_games > 0 else 0
            ppg_leaderboard_prep.append({
                "name": name, "ppg": ppg, "total_score": total_score, "total_games": total_games
            })
    ppg_leaderboard_prep.sort(key=lambda x: (-x["ppg"], -x["total_games"], x["name"].lower()))
    return ppg_leaderboard_prep[:top_n]

def prepare_active_tournaments_data(player_stats_raw, top_n=10, min_games_credit=5):
    # Assumes 'tournaments_played_min_5_games' key or similar exists from main script
    key_for_active_count = 'tournaments_played_min_5_games' 
    if not player_stats_raw: return []
    
    participation_data_prep = []
    for name, stats in player_stats_raw.items():
        tournaments_count = stats.get(key_for_active_count, 0) 
        if tournaments_count > 0:
            participation_data_prep.append({"player": name, "count": tournaments_count})
            
    participation_data_prep.sort(key=lambda x: (-x["count"], x["player"].lower()))
    return participation_data_prep[:top_n]


def generate_report():
    # Load data
    medal_table_raw = load_data('medal_table.json')
    player_stats_raw = load_data('player_overall_stats.json')
    berserk_table_raw = load_data('berserk_table.json')

    # Prepare data for template
    context = {
        "report_title": f"CNO Daily Crazyhouse Arena",
        "username": USERNAME_FOR_REPORT,
        "generation_time": datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        
        "medal_data": prepare_medal_data(medal_table_raw),
        
        "points_top_n": 10,
        "points_data": prepare_points_data(player_stats_raw, top_n=10),
        
        "berserk_top_n": 5, # Smaller for example
        "berserk_min_games": 30,
        "berserk_wr_data": [],
        "berserk_freq_data": [],

        "ppg_top_n": 5,
        "ppg_min_total_games": 30,
        "ppg_data": prepare_ppg_data(player_stats_raw, top_n=10, min_total_games=30),
        
        "active_top_n": 5,
        "active_min_games_credit": 3, # Must match key used in main script or logic
        "active_tournaments_data": prepare_active_tournaments_data(player_stats_raw, top_n=5, min_games_credit=3)
    }
    
    berserk_wr, berserk_freq = prepare_berserk_data(berserk_table_raw, top_n=context["berserk_top_n"], min_games=context["berserk_min_games"])
    context["berserk_wr_data"] = berserk_wr
    context["berserk_freq_data"] = berserk_freq

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('report_template.html')

    # Render HTML
    html_output = template.render(context)

    # Save HTML
    try:
        with open(HTML_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"HTML report generated: {HTML_OUTPUT_FILE}")
    except IOError:
        print(f"Error: Could not write HTML report to {HTML_OUTPUT_FILE}")

if __name__ == '__main__':
    generate_report()