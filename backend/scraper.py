import os 
from dotenv import load_dotenv
from github import Github, RateLimitExceededException, GithubExceptoin
import time

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
  raise Exception("GitHub Token not found! Please put the GITHUB_TOKEN")


g = Github(GITHUB_TOKEN)


def get_repo_data(repo):
  """
  Extract Relevant data from a repository object.
  """

  return {
    "full_name": repo.full_name,
    "language": repo.language,
    "stargazers_count": repo.stargazers_count,
    "forks_count": repo.forks_count,
    "description": repo.description or ""
  }


