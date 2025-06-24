import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- 1. Database Connection ---

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def fetch_data(driver):
  """
  Fetches nodes and relationships from neo4j and return them as Pandas Dataframes
  """

  with driver.session(database='neo4j') as session:
    #Fetch users
    user_query = "MATCH (u:User) RETURN u.login AS login, u.name AS name, u.bio AS bio"
    user_result = session.run(user_query)
    user_df = pd.DataFrame([record.data() for record in user_result])
    print(f'Fetched {len(user_df)} users.')

    #Fetch Repositories
    repos_query = "MATCH (r:Repository) RETURN r.full_name AS full_name, r.language AS language, r.description AS description, r.stargazers_count AS stargazers_count"
    repos_result = session.run(repos_query)
    repos_df = pd.DataFrame([record.data() for record in repos_result])
    print(f'Fetched {len(repos_df)} repositories.')

    #Fetch STARS relationship
    stars_query = "MATCH (u:User)-[:STARS]->(r:Repository) RETURN u.login AS source, r.full_name AS target"
    stars_result = session.run(stars_query)
    stars_df  = pd.DataFrame([record.data() for record in stars_result])
    print(f'Fetched {len(stars_df)} STARS relationship')
  
  return user_df, repos_df, stars_df