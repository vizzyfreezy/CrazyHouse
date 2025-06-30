import os
import json
import glicko2
from html_report_generator import load_json_file

SOURCE_DIR_GAMES = "detailed_games"
GLICKO_RD_CUTOFF = 80
DATA_DIR = "data_output_lastweek"


# TODO: logic to calculate based on date range and also use update ratings based on previous week
def load_all_detailed_games(games_dir):
    all_games = []
    if not os.path.exists(games_dir):
        return []
    for filename in os.listdir(games_dir):
        if filename.endswith(".json"):
            all_games.extend(
                load_json_file(os.path.join(games_dir, filename), default=[])
            )
    return all_games


def calculate_glicko2_rankings(all_games):
    if not all_games:
        return []
    players = {}
    player_names = {g["winner_name"] for g in all_games} | {
        g["loser_name"] for g in all_games
    }
    for name in player_names:
        players[name] = glicko2.Player()
    match_history = {
        name: {"ratings": [], "rds": [], "outcomes": []} for name in players
    }
    for game in all_games:
        winner_name, loser_name = game["winner_name"], game["loser_name"]
        winner_obj, loser_obj = players[winner_name], players[loser_name]
        match_history[winner_name]["ratings"].append(loser_obj.getRating())
        match_history[winner_name]["rds"].append(loser_obj.getRd())
        match_history[winner_name]["outcomes"].append(1.0)
        match_history[loser_name]["ratings"].append(winner_obj.getRating())
        match_history[loser_name]["rds"].append(winner_obj.getRd())
        match_history[loser_name]["outcomes"].append(0.0)
    for name, p_obj in players.items():
        history = match_history[name]
        if history["ratings"]:
            p_obj.update_player(history["ratings"], history["rds"], history["outcomes"])
    return {
        name: {"rating": p.getRating(), "rd": p.getRd()} for name, p in players.items()
    }


def calculate_new_glicko2_rankings(new_games, player_history):
    """
    Calculates Glicko-2 rankings for a set of games.

    Args:
        all_games (list): A list of dictionaries, where each dictionary
                          represents a game and has "winner_name" and "loser_name".
        players_data (dict, optional): A dictionary containing existing player data.
                                       The format is { "player_name": {"rating": R, "rd": RD, "vol": V} }.
                                       Defaults to {}.

    Returns:
        dict: A dictionary of all players with their updated rating and RD.
    """
    if not new_games:
        return player_history  # Return the initial data if there are no games

    # Use a dictionary to store the glicko2.Player objects
    players = {}

    # Get a set of all unique player names from the games played
    player_names_in_games = {g["winner_name"] for g in new_games} | {
        g["loser_name"] for g in new_games
    }

    # Load existing players or create new ones
    for name in player_names_in_games:
        if name in player_history:
            # If the player exists, load their data from the provided dictionary
            data = player_history[name]
            players[name] = glicko2.Player(
                rating=data.get("rating", 1500),
                rd=data.get("rd", 350),
                vol=data.get("vol", 0.06),
            )
        else:
            # If the player is new, create a default Player object
            players[name] = glicko2.Player()
    # Prepare lists to hold match results for each player
    match_history = {
        name: {"ratings": [], "rds": [], "outcomes": []}
        for name in player_names_in_games
    }

    # Populate the match history from the game results
    for game in new_games:
        winner_name, loser_name = game["winner_name"], game["loser_name"]

        # Ensure both players are in our `players` dictionary before proceeding
        if winner_name not in players or loser_name not in players:
            continue

        winner_obj, loser_obj = players[winner_name], players[loser_name]

        # Add the opponent's stats and the outcome to the winner's history
        match_history[winner_name]["ratings"].append(loser_obj.rating)
        match_history[winner_name]["rds"].append(loser_obj.rd)
        match_history[winner_name]["outcomes"].append(1.0)  # Winner gets a 1.0

        # Add the opponent's stats and the outcome to the loser's history
        match_history[loser_name]["ratings"].append(winner_obj.rating)
        match_history[loser_name]["rds"].append(winner_obj.rd)
        match_history[loser_name]["outcomes"].append(0.0)  # Loser gets a 0.0

    # Update the ratings for every player who competed
    for name, p_obj in players.items():
        history = match_history.get(name)
        if history and history["ratings"]:  # Check if the player had any games
            p_obj.update_player(history["ratings"], history["rds"], history["outcomes"])

    # Return the updated data for all players involved
    return {
        name: {"rating": p.rating, "rd": p.rd, "vol": p.vol}
        for name, p in players.items()
    }


def main():
    current_glicko_data = load_json_file(os.path.join(DATA_DIR, "glicko_rankings.json"))
    all_games = load_all_detailed_games(SOURCE_DIR_GAMES)
    glicko_rankings_all = calculate_new_glicko2_rankings(all_games, current_glicko_data)
    with open(os.path.join("data_output", "glicko_rankings.json"), "w") as f:
        json.dump(glicko_rankings_all, f, indent=4)


if __name__ == "__main__":
    main()
