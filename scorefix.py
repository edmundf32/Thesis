import csv
import requests
from github_repo_scorer import GitHubRepoScorer

# Cache for storing scores
scores_cache = {}

# Function to get scores, either from cache or by making an API call
def get_scores(owner, repo):
    repo_name = f"{owner}/{repo}"
    if repo_name in scores_cache:
        return scores_cache[repo_name]
    else:
        scorer = GitHubRepoScorer(owner, repo)
        scores = scorer.calculate_score()
        scores_cache[repo_name] = scores
        return scores

# Function to update scores in CSV
def update_scores_in_csv(csv_file_path):
    updated_rows = []
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            owner, repo = row[0].split('/')
            star_score, follower_score, community_score = get_scores(owner, repo)
            updated_row = row[:12] + [star_score, follower_score, community_score]
            updated_rows.append(updated_row)

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(updated_rows)

# Path to your CSV file
csv_file_path = 'javascript_github_repos_pull_requests_new_blank scores.csv'
update_scores_in_csv(csv_file_path)
