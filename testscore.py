from github_repo_scorer import GitHubRepoScorer

repo_name = 'chat2db/Chat2DB'
owner_repo = repo_name.split('/')
scorer = GitHubRepoScorer(owner_repo[0], owner_repo[1])
print('owner -', owner_repo[0])
print('repo  -', owner_repo[1])
print('score -', scorer.calculate_score())