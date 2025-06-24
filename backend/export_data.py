import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def export_data(driver):
  """
  Fetches data from neo4j and saves it to csv files
  """
  with driver.session(database='neo4j') as session:

    # Fetch users
    users_query = "MATCH (u:User) RETURN u.login AS login, u.name AS name, u.bio AS bio"
    users_result = session.run(users_query)
    users_df = pd.DataFrame([record.data() for record in users_result])
    users_df.to_csv("users.csv", index=False)
    print(f"Exported {len(users_df)} users to users.csv")

    #Fetch Repositories
    repos_query = "MATCH (r:Repository) RETURN r.full_name AS full_name, r.language AS language, r.description AS description, r.stargazers_count AS stargazers_count"
    repos_result = session.run(repos_query)
    repos_df = pd.DataFrame([record.data() for record in repos_result])
    repos_df.to_csv("repos.csv", index=False)
    print(f"Exported {len(repos_df)} repositories to repos.csv")

    #Fetch STARS relationship
    stars_query = "MATCH (u:User)-[:STARS]->(r:Repository) RETURN u.login AS source, r.full_name AS target"
    stars_result = session.run(stars_query)
    stars_df  = pd.DataFrame([record.data() for record in stars_result])
    stars_df.to_csv("stars.csv", index=False)
    print(f"Exported {len(stars_df)} STARTS relationships to stars.csv")



if __name__ == "__main__":
  driver = None
  try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("--- Connected to Neo4j. Exporting data... ---")
    export_data(driver)
    print("--- Data export complete! ---")
  except Exception as e:
    print(f"An error occured: {e}")
  finally:
    if driver:
      driver.close()