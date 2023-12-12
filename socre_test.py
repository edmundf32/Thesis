import requests

class GitHubRepoScorer:
    def __init__(self, owner, repo):
        self.owner = owner
        self.repo = repo
        self.repo_url = f"https://api.github.com/repos/chat2db/Chat2DB"
        self.community_profile_url = f"https://api.github.com/repos/chat2db/Chat2DB/community/profile"

    def get_repo_details(self):
        response = requests.get(self.repo_url)
        if response.status_code == 200:
            repo_details = response.json()
            stars = repo_details.get('stargazers_count', 0)
            followers = repo_details.get('subscribers_count', 0)
            return stars, followers
        return 0, 0

    def get_community_profile_count(self):
        response = requests.get(self.community_profile_url)
        if response.status_code == 200:
            profile = response.json()
            files = profile.get('files', {})
            count = sum(1 for key in ['code_of_conduct', 'contributing', 'license', 'readme', 'issue_template', 'pull_request_template'] if key in files)
            return count
        return 0

    def calculate_score(self):
        stars, followers = self.get_repo_details()
        community_items = self.get_community_profile_count()

        return stars, followers, community_items

