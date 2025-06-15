import json
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime, timezone

# --- CONFIGURATION (OLD TABLES - GOLD STANDARD) ---
DATA_DIR = "data_output"
LAST_WEEK_DIR = "data_output_lastweek"
TABLE_LIMITS = {"medals": 10, "ppg": 10, "berserk": 10, "iron_man": 10, "power_rankings": 15}
PPG_MIN_TOTAL_GAMES = 20
BERSERK_MIN_GAMES = 10

# --- CONFIGURATION (NEW TABLES) ---
SOURCE_DIR_GAMES = "detailed_games"
POWER_RANKING_ITERATIONS = 100
POWER_RANKING_INITIAL_SCORE = 1000
POWER_RANKING_TRANSFER_FACTOR = 0.03

# --- GLOBAL CONFIG ---
TEMPLATE_DIR = "templates"
TEMPLATE_NAME = "report_template.html"
OUTPUT_FILENAME = "Crazyhouse_Weekly_Report.html"
USERNAME = "BayorMiller"

# =================================================================================
# SECTION 1: "GOLD STANDARD" LOGIC - RESTORED AND PRESERVED
# =================================================================================

def load_json_file(filepath, default=None):
    if default is None: default = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default

def get_last_week_ranks(last_week_data, key_func, id_field='player'):
    if not last_week_data: return {}
    last_week_list = list(last_week_data.values())
    last_week_list.sort(key=key_func, reverse=True)
    return {item.get(id_field): rank + 1 for rank, item in enumerate(last_week_list)}

def add_rank_trend(ranked_list, last_week_ranks, id_field='player'):
    for i, item in enumerate(ranked_list):
        current_rank, player_id = i + 1, item[id_field]
        last_rank = last_week_ranks.get(player_id)
        if last_rank is None: item['rank_change'] = 'new'
        else: item['rank_change'] = last_rank - current_rank
    return ranked_list

def process_medal_data(current_medals, last_week_medals):
    processed = [{'player': p, 'score': (m.get('gold',0)*3 + m.get('silver',0)*2 + m.get('bronze',0)), **m} for p, m in current_medals.items()]
    processed.sort(key=lambda x: (x['score'], x['gold'], x['silver']), reverse=True)
    lw_key_func = lambda x: ((x.get('gold',0)*3 + x.get('silver',0)*2 + x.get('bronze',0)), x.get('gold',0), x.get('silver',0))
    lw_ranks = get_last_week_ranks(last_week_medals, lw_key_func, id_field='name')
    processed = add_rank_trend(processed, lw_ranks, id_field='player')
    return processed

def process_ppg_data(overall_stats, last_week_stats):
    eligible = {p: s for p, s in overall_stats.items() if s.get('total_games', 0) >= PPG_MIN_TOTAL_GAMES}
    processed = [{'player': p, 'ppg': s['total_score']/s['total_games'] if s['total_games']>0 else 0, **s} for p, s in eligible.items()]
    processed.sort(key=lambda x: x['ppg'], reverse=True)
    lw_eligible = {p: s for p, s in last_week_stats.items() if s.get('total_games', 0) >= PPG_MIN_TOTAL_GAMES}
    lw_key_func = lambda s: (s.get('total_score',0) / s.get('total_games',1))
    lw_ranks = get_last_week_ranks(lw_eligible, lw_key_func, id_field='name')
    processed = add_rank_trend(processed, lw_ranks, id_field='player')
    return processed

def process_berserk_data(berserk_table, last_week_berserk):
    eligible = {p: s for p, s in berserk_table.items() if s.get('berserk_games', 0) >= BERSERK_MIN_GAMES}
    processed = []
    for player, stats in eligible.items():
        freq = stats['berserk_games']/stats['total_games_in_podium_summary'] if stats['total_games_in_podium_summary']>0 else 0
        win_rate = stats['berserk_wins']/stats['berserk_games'] if stats['berserk_games']>0 else 0
        processed.append({'player': player, 'freq': freq, 'win_rate': win_rate, **stats})
    processed.sort(key=lambda x: x['freq'], reverse=True)
    lw_eligible = {p: s for p, s in last_week_berserk.items() if s.get('berserk_games', 0) >= BERSERK_MIN_GAMES}
    lw_key_func = lambda s: (s.get('berserk_games',0) / s.get('total_games_in_podium_summary',1))
    lw_ranks = get_last_week_ranks(lw_eligible, lw_key_func, id_field='name')
    processed = add_rank_trend(processed, lw_ranks, id_field='player')
    return processed

def process_iron_man_data(overall_stats, last_week_stats):
    processed = [{'player': p, **s} for p, s in overall_stats.items()]
    processed.sort(key=lambda x: x.get('total_games', 0), reverse=True)
    lw_key_func = lambda s: s.get('total_games', 0)
    lw_ranks = get_last_week_ranks(last_week_stats, lw_key_func, id_field='name')
    processed = add_rank_trend(processed, lw_ranks, id_field='player')
    return processed

# =================================================================================
# SECTION 2: NEW FEATURES LOGIC - ADDED SEPARATELY
# =================================================================================

def load_all_detailed_games(games_dir):
    all_games = []
    for filename in os.listdir(games_dir):
        if filename.endswith('.json'):
            all_games.extend(load_json_file(os.path.join(games_dir, filename), default=[]))
    return all_games

def calculate_power_rankings(all_games):
    if not all_games: return []
    players = {g['winner_name'] for g in all_games} | {g['loser_name'] for g in all_games}
    power_scores = {p: float(POWER_RANKING_INITIAL_SCORE) for p in players}
    total_score_pool = len(players) * POWER_RANKING_INITIAL_SCORE
    for _ in range(POWER_RANKING_ITERATIONS):
        new_scores = power_scores.copy()
        for g in all_games:
            transfer = POWER_RANKING_TRANSFER_FACTOR * power_scores[g['loser_name']]
            new_scores[g['winner_name']] += transfer
            new_scores[g['loser_name']] -= transfer
        current_total = sum(new_scores.values())
        if current_total == 0: continue
        scaling_factor = total_score_pool / current_total
        power_scores = {p: s * scaling_factor for p, s in new_scores.items()}
    ranked_list = [{'player': p, 'score': s} for p, s in power_scores.items()]
    ranked_list.sort(key=lambda x: x['score'], reverse=True)
    return ranked_list

def calculate_inclusive_stats(all_games):
    if not all_games: return {"giant_killer": None, "marathon_man": None}
    biggest_upset = max(all_games, key=lambda g: g['loser_rating'] - g['winner_rating'], default=None)
    if biggest_upset and (biggest_upset['loser_rating'] - biggest_upset['winner_rating'] <= 0): biggest_upset = None
    longest_game = max(all_games, key=lambda g: g['moves'], default=None)
    return {"giant_killer": biggest_upset, "marathon_man": longest_game}

# =================================================================================
# SECTION 3: THE UNIFIED MAIN FUNCTION
# =================================================================================

def main():
    print("--- CNO Stats Engine V12: Generating Final Unified Report ---")

    # --- PIPELINE A: GOLD STANDARD FOR OLD TABLES ---
    print("Loading pre-processed data for standard tables...")
    current_medals = load_json_file(os.path.join(DATA_DIR, "medal_table.json"))
    current_overall = load_json_file(os.path.join(DATA_DIR, "player_overall_stats.json"))
    current_berserk = load_json_file(os.path.join(DATA_DIR, "berserk_table.json"))
    
    last_week_medals = load_json_file(os.path.join(LAST_WEEK_DIR, "medal_table.json"))
    last_week_overall = load_json_file(os.path.join(LAST_WEEK_DIR, "player_overall_stats.json"))
    last_week_berserk = load_json_file(os.path.join(LAST_WEEK_DIR, "berserk_table.json"))
    
    # Gold standard requires adding 'name' key for uniform processing in helper
    for name, data in last_week_overall.items(): data['name'] = name
    for name, data in last_week_medals.items(): data['name'] = name
    for name, data in last_week_berserk.items(): data['name'] = name

    print("Processing standard tables with trend analysis...")
    medal_data = process_medal_data(current_medals, last_week_medals)
    ppg_data = process_ppg_data(current_overall, last_week_overall)
    berserk_data = process_berserk_data(current_berserk, last_week_berserk)
    iron_man_data = process_iron_man_data(current_overall, last_week_overall)

    # --- PIPELINE B: NEW FEATURES ---
    print("Loading detailed game data for advanced stats...")
    all_games = load_all_detailed_games(SOURCE_DIR_GAMES)
    power_rankings = calculate_power_rankings(all_games)
    inclusive_stats = calculate_inclusive_stats(all_games)

    # --- FINAL CONTEXT FOR TEMPLATE ---
    context = {
        'generation_date': datetime.now().strftime("%B %d, %Y at %I:%M %p"), 'username': USERNAME, 'table_limits': TABLE_LIMITS, 'ppg_min_total_games': PPG_MIN_TOTAL_GAMES,
        'power_rankings': power_rankings[:TABLE_LIMITS['power_rankings']], 
        'inclusive_stats': inclusive_stats,
        'medal_data': medal_data[:TABLE_LIMITS['medals']], 
        'ppg_data': ppg_data[:TABLE_LIMITS['ppg']],
        'berserk_data': berserk_data[:TABLE_LIMITS['berserk']], 
        'iron_man_data': iron_man_data[:TABLE_LIMITS['iron_man']],
    }

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template(TEMPLATE_NAME)
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f: f.write(template.render(context))
    print(f"\n✅ Success! Definitive report generated: {os.path.abspath(OUTPUT_FILENAME)}")
    
if __name__ == "__main__":
    main()