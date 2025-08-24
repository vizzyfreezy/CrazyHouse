import pandas as pd
import argparse
from sklearn.cluster import KMeans

def cluster_players_by_performance(n_clusters: int):
    """
    Clusters players into n groups based on their Avg Score from the simulation.

    Args:
        n_clusters: The number of clusters to create.
    """
    try:
        player_report = pd.read_csv('player_performance_report.csv', index_col=0)
    except FileNotFoundError:
        print("!!! ERROR: player_performance_report.csv not found.")
        print("Please run 'simulate_teambattle.py' first to generate the report.")
        return

    # Reshape the data for clustering
    X = player_report[['Avg Score']].values

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    player_report['Cluster'] = kmeans.fit_predict(X)

    # Sort players by 'Avg Score' for ordered output
    sorted_players = player_report.sort_values(by='Avg Score', ascending=False)

    # Calculate the mean 'Avg Score' for each cluster
    cluster_means = sorted_players.groupby('Cluster')['Avg Score'].mean().sort_values(ascending=False)

    # Print the clusters in order of their mean 'Avg Score'
    print(f"\n--- Player Strength Clusters (based on Avg Score) ---")
    for cluster_id in cluster_means.index:
        cluster = sorted_players[sorted_players['Cluster'] == cluster_id]
        print(f"--- Cluster (Avg Score: {cluster_means[cluster_id]:.2f}) ---")
        for player_name, row in cluster.iterrows():
            avg_score = row['Avg Score']
            rating = row['Rating']
            print(f"  {player_name} (Avg Score: {avg_score:.2f}, Rating: {rating})")
        print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster players based on their simulated performance.')
    parser.add_argument('n_clusters', type=int, help='The number of clusters to create.')
    args = parser.parse_args()

    cluster_players_by_performance(args.n_clusters)
