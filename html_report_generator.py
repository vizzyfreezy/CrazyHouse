import json
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

# --- CONFIGURATION ---
# These paths should be relative to where you run the script from.
# Assuming your folder structure is:
# /CrazyHouse
#   - report_generator.py
#   - /data_output
#   - /data_output_lastweek
#   - /templates

DATA_DIR = "data_output"
LAST_WEEK_DIR = "data_output_lastweek"
TEMPLATE_DIR = "templates"
TEMPLATE_NAME = "report_template.html"
OUTPUT_FILENAME = "Crazyhouse_Weekly_Report.html"

# Report-specific settings
REPORT_TITLE = "CNO Crazyhouse Weekly Report"
USERNAME = "BayorMiller"
PPG_MIN_TOTAL_GAMES = 20
BERSERK_MIN_GAMES = 10

def load_json_file(filepath):
    """Safely loads a JSON file, returning an empty dict if it doesn't exist."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load or parse {filepath}. Skipping.")
        return {}

def process_medal_data(medal_table):
    """Calculates podium score and sorts players."""
    processed = []
    for player, medals in medal_table.items():
        score = (medals.get('gold', 0) * 3) + \
                (medals.get('silver', 0) * 2) + \
                (medals.get('bronze', 0) * 1)
        
        player_data = {
            'player': player,
            'score': score,
            'gold': medals.get('gold', 0),
            'silver': medals.get('silver', 0),
            'bronze': medals.get('bronze', 0)
        }
        processed.append(player_data)
    
    # Sort by score (desc), then gold (desc), then silver (desc) as tie-breakers
    processed.sort(key=lambda x: (x['score'], x['gold'], x['silver']), reverse=True)
    return processed

def process_ppg_data(overall_stats, min_games):
    """Calculates PPG for players with a minimum number of games."""
    processed = []
    for player, stats in overall_stats.items():
        if stats.get('total_games', 0) >= min_games:
            total_games = stats['total_games']
            ppg = (stats['total_score'] / total_games) if total_games > 0 else 0
            processed.append({
                'name': player,
                'ppg': ppg,
                'total_score': stats['total_score'],
                'total_games': total_games
            })
    
    processed.sort(key=lambda x: x['ppg'], reverse=True)
    return processed

def process_berserk_data(berserk_table, min_berserk_games):
    """Creates the combined Berserk table."""
    processed = []
    for player, stats in berserk_table.items():
        bz_games = stats.get('berserk_games', 0)
        if bz_games >= min_berserk_games:
            total_games_podium = stats.get('total_games_in_podium_summary', 0)
            bz_wins = stats.get('berserk_wins', 0)
            
            freq = (bz_games / total_games_podium) if total_games_podium > 0 else 0
            win_rate = (bz_wins / bz_games) if bz_games > 0 else 0
            
            processed.append({
                'name': player,
                'berserk_freq': freq,
                'win_rate': win_rate,
                'berserk_wins': bz_wins,
                'berserk_losses': bz_games - bz_wins
            })
            
    # Sort by Berserk Frequency
    processed.sort(key=lambda x: x['berserk_freq'], reverse=True)
    return processed

def process_iron_man_data(overall_stats):
    """Ranks players by total games played."""
    processed = []
    for player, stats in overall_stats.items():
        processed.append({
            'player': player,
            'total_games': stats.get('total_games', 0)
        })
    
    processed.sort(key=lambda x: x['total_games'], reverse=True)
    return processed

def process_ppg_trend_data(current_stats, last_week_stats):
    """Calculates the change in PPG from last week."""
    processed = []
    for player, c_stats in current_stats.items():
        if player in last_week_stats:
            l_stats = last_week_stats[player]
            
            c_games = c_stats.get('total_games', 0)
            l_games = l_stats.get('total_games', 0)

            # Only calculate trend for players who are still active
            if c_games > 0 and l_games > 0:
                current_ppg = c_stats['total_score'] / c_games
                last_week_ppg = l_stats['total_score'] / l_games
                ppg_change = current_ppg - last_week_ppg
                
                processed.append({
                    'player': player,
                    'current_ppg': current_ppg,
                    'ppg_change': ppg_change
                })

    # Sort by the biggest improvement
    processed.sort(key=lambda x: x['ppg_change'], reverse=True)
    return processed

def main():
    """Main function to generate the report."""
    print("--- CNO Stats Engine: Starting Report Generation ---")

    # 1. Load data from files
    print("Loading data files...")
    current_medals = load_json_file(os.path.join(DATA_DIR, "medal_table.json"))
    current_overall = load_json_file(os.path.join(DATA_DIR, "player_overall_stats.json"))
    current_berserk = load_json_file(os.path.join(DATA_DIR, "berserk_table.json"))
    
    last_week_overall = load_json_file(os.path.join(LAST_WEEK_DIR, "player_overall_stats.json"))
    
    # 2. Process data for the template
    print("Processing stats...")
    medal_data = process_medal_data(current_medals)
    ppg_data = process_ppg_data(current_overall, PPG_MIN_TOTAL_GAMES)
    berserk_data = process_berserk_data(current_berserk, BERSERK_MIN_GAMES)
    iron_man_data = process_iron_man_data(current_overall)
    ppg_trend_data = process_ppg_trend_data(current_overall, last_week_overall)

    # 3. Prepare context for Jinja2 template
    context = {
        'report_title': REPORT_TITLE,
        'generation_date': datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        'username': USERNAME,
        
        # Data for tables
        'medal_data': medal_data,
        'ppg_data': ppg_data,
        'berserk_data': berserk_data,
        'iron_man_data': iron_man_data,
        'ppg_trend_data': ppg_trend_data,

        # Config variables needed in template
        'ppg_min_total_games': PPG_MIN_TOTAL_GAMES
    }

    # 4. Render the HTML report
    print("Rendering HTML report...")
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_NAME)
    
    output_html = template.render(context)

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(output_html)
        
    print(f"\n✅ Success! Report generated: {os.path.abspath(OUTPUT_FILENAME)}")
    print("--- CNO Stats Engine: Finished ---")


if __name__ == "__main__":
    main()