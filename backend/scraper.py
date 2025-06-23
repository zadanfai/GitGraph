import os 
from dotenv import load_dotenv
from github import Github, RateLimitExceededException, GithubExceptoin
from neo4j import GraphDatabase
import time

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
  raise Exception("GitHub Token not found! Please put the GITHUB_TOKEN")

g = Github(GITHUB_TOKEN)

# Neo4j authentication
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise Exception("Neo4j credentials not found in .env file.")


def get_repo_data(repo):
  """Extract Relevant data from a repository object."""
  return {
    "full_name": repo.full_name,
    "language": repo.language,
    "stargazers_count": repo.stargazers_count,
    "forks_count": repo.forks_count,
    "description": repo.description or ""
  }


def get_user_data(user):
  """Extracts relevant data from a user object"""
  return {
    "login": user.login,
    "name": user.name,
    "bio": user.bio
  }


def scrape_github_data(seed_repo_name, max_repos=50, max_stargazers=50):
  """
  Scrapes Github data starting from a seed repository

  :param seed_repo_name: The starting repository (e.g., "tensorflow/tensorflow").
  :param max_repos: The max number of new repositories to explore.
  :param max_stargazers: The max number of stargazers to explore per repository.
  """
  
  repos_to_scrape = {seed_repo_name}
  scraped_repos = set()

  scraped_data = {
    'repos': {},
    'users': {},
    'stars': []
  }

  while repos_to_scrape and len(scraped_repos) < max_repos:
    current_repo_name = repos_to_scrape.pop()

    if current_repo_name in scraped_repos:
      continue

    try:
      print(f"--- Scraping repository: {current_repo_name} ---")
      repo = g.get_repo(current_repo_name)

      #? Saved repo data
      scraped_data['repos'][repo.full_name] = get_repo_data(repo)
      scraped_repos.append(repo.full_name)

      #? Explore stargazers of this repo
      print(f" Fetching stargazers...")
      stargazers = repo.get_stargazers().get_page(0) #? Get first page of stargazers

      for i, user in enumerate(stargazers):
        if i >= max_stargazers:
          break
        
        #? Save user data
        print(f"     - Found user: {user.login}")
        if user.login not in scraped_data['users']:
          scraped_data['users'][user.login] = get_user_data(user)
        
        #? Record the star relationship
        scraped_data['stars'].append((user.login, repo.full_name))

        #? Add repos starred by this user to our list to scrape
        #! this is how we discover new, related repositories
        user_starred_repos = user.get_starred().get_page(0)
        for starred_repo in user_starred_repos:
          if starred_repo.full_name not in scraped_repos:
            repos_to_scrape.add(starred_repo.full_name)
        
      print(f'Rate limit remaining: {g.rate_limiting[0]}')
    
    except RateLimitExceededException:
      print("!!! Rate limit exceeded. Waiting for 60 seconds. !!!")
      time.sleep(60)
      repos_to_scrape.add(current_repo_name)
    except GithubExceptoin as e:
      print(f"!!! An error occurred for repo {current_repo_name}: {e} !!!")
    except Exception as e:
      print(f" Unexpeceted error occured: {e}")
  

  print(f"\n--- Scrping Complete ---")
  print(f"Total Repos Found: {len(scraped_data['repos'])}")
  print(f"Total Users Found: {len(scraped_data['users'])}")
  print(f"Total Star Relationships Found: {len(scraped_data['stars'])}")


  return scraped_data


def ingest_data_to_neo4j(driver, data):
  """Writes the scraped data into the Neo4j database."""
  # Create uniqueness constraints for faster lookups and data integrity
  with driver.session(database="neo4j") as session:
      session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Repository) REQUIRE r.full_name IS UNIQUE")
      session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.login IS UNIQUE")

  # Ingest repositories
  with driver.session(database="neo4j") as session:
      for repo_name, repo_data in data["repos"].items():
          session.execute_write(
              lambda tx: tx.run("""
                  MERGE (r:Repository {full_name: $full_name})
                  SET r.language = $language,
                      r.stargazers_count = $stargazers_count,
                      r.forks_count = $forks_count,
                      r.description = $description
              """, **repo_data)
          )
  print("Ingested all Repository nodes.")

  # Ingest users
  with driver.session(database="neo4j") as session:
      for user_login, user_data in data["users"].items():
          session.execute_write(
              lambda tx: tx.run("""
                  MERGE (u:User {login: $login})
                  SET u.name = $name,
                      u.bio = $bio
              """, **user_data)
          )
  print("Ingested all User nodes.")

  # Ingest star relationships
  with driver.session(database="neo4j") as session:
      for user_login, repo_name in data["stars"]:
          session.execute_write(
              lambda tx: tx.run("""
                  MATCH (u:User {login: $user_login})
                  MATCH (r:Repository {full_name: $repo_name})
                  MERGE (u)-[:STARS]->(r)
              """, user_login=user_login, repo_name=repo_name)
          )
          
  print("Ingested all STARS relationships.")


if __name__ == "__main__":
  seed_repository = "tensorflow/tensorflow"
  final_data = scrape_github_data(seed_repository)

  # import json
  # print(json.dumps(final_data, indent=2))

  driver = None
  try:
    # Create a driver instance
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity() 
    print("\n--- Connecting to Neo4j and starting ingestion... ---")
    ingest_data_to_neo4j(driver, final_data)
    print("--- Data ingestion complete! ---")
  except Exception as e:
    print(f'An error occured during Neo4j ingestion: {e}')
  finally:
    if driver:
      driver.close()