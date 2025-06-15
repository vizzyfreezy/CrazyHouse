import json
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR = "data_output"
LAST_WEEK_DIR = "data_output_lastweek"
TEMPLATE_DIR = "templates"
TEMPLATE_NAME = "report_template.html"
OUTPUT_FILENAME = "Crazyhouse_Weekly_Report.html"

REPORT_TITLE = "CNO Crazyhouse Weekly Report"
USERNAME = "BayorMiller"

# --- NEW: CONTROL TABLE LENGTHS HERE ---
TABLE_LIMITS = {
    "medals": 10,
    "ppg": 10,
    "berserk": 10,
    "iron_man": 10
}

# --- MINIMUM GAME REQUIREMENTS ---
PPG_MIN_TOTAL_GAMES = 20
BERSERK_MIN_GAMES = 10


def load_json_file(filepath):
    """Safely loads a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_last_week_ranks(last_week_data, key_func, id_field='player'):
    """Helper function to create a map of player -> last week's rank."""
    if not last_week_data:
        return {}
    
    # Filter and sort last week's data to establish ranks
    last_week_list = [v for v in last_week_data.values()]
    last_week_list.sort(key=key_func, reverse=True)
    
    return {item[id_field]: rank + 1 for rank, item in enumerate(last_week_list)}

def add_rank_trend(ranked_list, last_week_ranks, id_field='player'):
    """Adds a 'rank_change' field to each item in the ranked list."""
    for i, item in enumerate(ranked_list):
        current_rank = i + 1
        player_id = item[id_field]
        last_rank = last_week_ranks.get(player_id)
        
        if last_rank is None:
            item['rank_change'] = 'new'
        else:
            item['rank_change'] = last_rank - current_rank # Positive is UP, Negative is DOWN
    return ranked_list

def process_medal_data(current_medals, last_week_medals):
    """Calculates podium score and sorts players, including rank trends."""
    processed = []
    for player, medals in current_medals.items():
        score = (medals.get('gold', 0) * 3) + (medals.get('silver', 0) * 2) + (medals.get('bronze', 0) * 1)
        processed.append({'player': player, 'score': score, 'gold': medals.get('gold', 0), 'silver': medals.get('silver', 0), 'bronze': medals.get('bronze', 0)})
    
    processed.sort(key=lambda x: (x['score'], x['gold'], x['silver']), reverse=True)
    
    # Get last week's ranks
    lw_key_func = lambda x: ((x.get('gold',0)*3 + x.get('silver',0)*2 + x.get('bronze',0)), x.get('gold',0), x.get('silver',0))
    lw_ranks = get_last_week_ranks({p: m for p, m in last_week_medals.items()}, lw_key_func, id_field='name') # Assuming player name is the key
    
    processed = add_rank_trend(processed, {p: r for p, r in lw_ranks.items()}, id_field='player')
    return processed[:TABLE_LIMITS['medals']]


def process_ppg_data(overall_stats, last_week_stats):
    """Calculates PPG and includes rank trends."""
    eligible_players = {p: s for p, s in overall_stats.items() if s.get('total_games', 0) >= PPG_MIN_TOTAL_GAMES}
    processed = []
    for player, stats in eligible_players.items():
        total_games = stats['total_games']
        ppg = (stats['total_score'] / total_games) if total_games > 0 else 0
        processed.append({'name': player, 'ppg': ppg, 'total_score': stats['total_score'], 'total_games': total_games})
    
    processed.sort(key=lambda x: x['ppg'], reverse=True)
    
    lw_eligible_players = {p: s for p, s in last_week_stats.items() if s.get('total_games', 0) >= PPG_MIN_TOTAL_GAMES}
    lw_key_func = lambda s: (s['total_score'] / s['total_games']) if s.get('total_games',0) > 0 else 0
    lw_ranks = get_last_week_ranks(lw_eligible_players, lw_key_func, id_field='name')

    processed = add_rank_trend(processed, lw_ranks, id_field='name')
    return processed[:TABLE_LIMITS['ppg']]

def process_berserk_data(berserk_table, last_week_berserk):
    """Creates the combined Berserk table with rank trends."""
    eligible_players = {p: s for p, s in berserk_table.items() if s.get('berserk_games', 0) >= BERSERK_MIN_GAMES}
    processed = []
    for player, stats in eligible_players.items():
        bz_games = stats['berserk_games']
        total_games_podium = stats.get('total_games_in_podium_summary', 0)
        bz_wins = stats.get('berserk_wins', 0)
        freq = (bz_games / total_games_podium) if total_games_podium > 0 else 0
        win_rate = (bz_wins / bz_games) if bz_games > 0 else 0
        processed.append({'name': player, 'berserk_freq': freq, 'win_rate': win_rate, 'berserk_wins': bz_wins, 'berserk_losses': bz_games - bz_wins})

    processed.sort(key=lambda x: x['berserk_freq'], reverse=True)
    
    lw_eligible_players = {p: s for p, s in last_week_berserk.items() if s.get('berserk_games', 0) >= BERSERK_MIN_GAMES}
    lw_key_func = lambda s: (s['berserk_games'] / s['total_games_in_podium_summary']) if s.get('total_games_in_podium_summary',0) > 0 else 0
    lw_ranks = get_last_week_ranks(lw_eligible_players, lw_key_func, id_field='name')
    
    processed = add_rank_trend(processed, lw_ranks, id_field='name')
    return processed[:TABLE_LIMITS['berserk']]

def process_iron_man_data(overall_stats, last_week_stats):
    """Ranks players by total games played with trends."""
    processed = [{'player': player, 'total_games': stats.get('total_games', 0)} for player, stats in overall_stats.items()]
    processed.sort(key=lambda x: x['total_games'], reverse=True)
    
    lw_key_func = lambda s: s.get('total_games', 0)
    lw_ranks = get_last_week_ranks(last_week_stats, lw_key_func, id_field='name')
    
    processed = add_rank_trend(processed, lw_ranks, id_field='player')
    return processed[:TABLE_LIMITS['iron_man']]

def main():
    print("--- CNO Stats Engine V2: Starting Report Generation ---")

    # 1. Load all necessary data
    print("Loading current and last week's data files...")
    current_medals = load_json_file(os.path.join(DATA_DIR, "medal_table.json"))
    current_overall = load_json_file(os.path.join(DATA_DIR, "player_overall_stats.json"))
    current_berserk = load_json_file(os.path.join(DATA_DIR, "berserk_table.json"))
    
    last_week_medals = load_json_file(os.path.join(LAST_WEEK_DIR, "medal_table.json"))
    last_week_overall = load_json_file(os.path.join(LAST_WEEK_DIR, "player_overall_stats.json"))
    last_week_berserk = load_json_file(os.path.join(LAST_WEEK_DIR, "berserk_table.json"))

    # Add player names to stats dicts to make them uniform for helper functions
    for name, data in last_week_overall.items(): data['name'] = name
    for name, data in last_week_medals.items(): data['name'] = name
    for name, data in last_week_berserk.items(): data['name'] = name

    # 2. Process data for the template, now including trend calculation
    print("Processing stats and calculating rank trends...")
    medal_data = process_medal_data(current_medals, last_week_medals)
    ppg_data = process_ppg_data(current_overall, last_week_overall)
    berserk_data = process_berserk_data(current_berserk, last_week_berserk)
    iron_man_data = process_iron_man_data(current_overall, last_week_overall)

    # 3. Prepare context for Jinja2 template
    context = {
        'report_title': REPORT_TITLE,
        'generation_date': datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        'username': USERNAME,
        'table_limits': TABLE_LIMITS,
        'ppg_min_total_games': PPG_MIN_TOTAL_GAMES,
        'medal_data': medal_data,
        'ppg_data': ppg_data,
        'berserk_data': berserk_data,
        'iron_man_data': iron_man_data,
    }

    # 4. Render the HTML report
    print("Rendering final HTML report...")
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_NAME)
    output_html = template.render(context)

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(output_html)
        
    print(f"\n✅ Success! Report generated: {os.path.abspath(OUTPUT_FILENAME)}")
    print("--- CNO Stats Engine V2: Finished ---")

if __name__ == "__main__":
    main()