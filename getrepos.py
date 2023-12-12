import requests
import psycopg2
import csv
from datetime import datetime
from github_repo_scorer import GitHubRepoScorer



def get_top_js_repositories():
    start_date = "2023-06-01"
    end_date = "2023-10-31"
    url = "https://api.github.com/search/repositories"
    
    date_range = f"{start_date}..{end_date}"
    query = {
        "q": f"language:python created:{date_range}",
        "sort": "stars",
        "order": "desc",
        "per_page": 100
    }

    response = requests.get(url, params=query)
    if response.status_code == 200:
        repos = [repo['full_name'] for repo in response.json()['items']]
        for repo in repos:
            print(repo)
        return repos
    else:
        return None

# Example usage


def find_repo_ids_and_pull_requests(repo_names):
    results = {}
    try:
        conn = psycopg2.connect(
            dbname="postgres", 
            user="postgres", 
            password="gobshite", 
            host="dissertation-instance-1.chopdurqdigf.eu-west-1.rds.amazonaws.com"
        )
        cur = conn.cursor()

        for repo_name in repo_names:
            start_time = datetime.now()  # Record start time
            print(f"Start processing: {repo_name} at {start_time}")
            
            cur.execute("SELECT a.repo_id FROM archive a WHERE a.repo_name = %s LIMIT 1;", (repo_name,))
            repo_id_result = cur.fetchone()
            repo_id = repo_id_result[0] if repo_id_result else None

            if repo_id:
                cur.execute("""
                SELECT a."type", pr."action", pr.user_type, pr.merged, pr."comments", 
                pr.state, pr.changed_files, pr.deletions, pr.additions, pr.review_comments, 
                pr.author_association FROM archive a
                INNER JOIN pullrequest pr ON a.payload_id = pr.id
                WHERE a.repo_id = %s
                ORDER BY a.repo_id DESC;
                """, (repo_id,))
                pr_data = cur.fetchall()
                results[repo_name] = {
                    'repo_id': repo_id,
                    'pull_requests': pr_data
                }
                
            end_time = datetime.now()  # Record end time
            elapsed_time = end_time - start_time
            print(f"Finished processing: {repo_name} at {end_time} (Elapsed Time: {elapsed_time})")


        cur.close()
        conn.close()

    except psycopg2.Error as e:
        print(f"Database error: {e}")

    return results

def write_to_csv(data):
    with open('python_github_repos_pull_requests_new.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Remove 'Comments' from the header
        writer.writerow(['Repo Name', 'Repo ID', 'Type', 'Action', 'User Type', 'Merged', 'State', 'Changed Files', 'Deletions', 'Additions', 'Review Comments', 'Author Association', 'Star Score', 'Follower Score', 'Community Score'])

        for repo_name, repo_info in data.items():
            print('Processing PR for repo:', repo_name)
            
            owner_repo = repo_name.split('/')
            scorer = GitHubRepoScorer(owner_repo[0], owner_repo[1])
            star_score, follower_score, community_score = scorer.calculate_score()  # Unpack the scores

            for pr in repo_info['pull_requests']:
                # Adjust the data row to exclude 'Comments'
                writer.writerow([repo_name, repo_info['repo_id'], pr[0], pr[1], pr[2], pr[3], pr[5], pr[6], pr[7], pr[8], pr[9], pr[10], star_score, follower_score, community_score])



                



# Main execution
top_js_repos = get_top_js_repositories()
if top_js_repos:
    data = find_repo_ids_and_pull_requests(top_js_repos)
    write_to_csv(data)
    print("Data written to 'python_github_repos_pull_requests.csv'")
else:
    print("Failed to fetch repositories from GitHub")
