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



# --- 2. Data Preprocessing & Feature Engineering ---

def preprocess_data(users_df, repos_df):

  #? Missing Text Data
  users_df['bio'].fillna('', inplace=True)
  repos_df['description'].fillna('', inplace=True)
  repos_df['language'].fillna('', inplace=True)

  #? Missing numerical data
  repos_df['stargazers_count'].fillna(0, inplace=True)

  #! Feature Engineering
  #? TF-IDF
  all_text = pd.concat([users_df['bio'], repos_df['description'], repos_df['language']], ignore_index=True)
  vectorizer = TfidfVectorizer(max_features=128)
  vectorizer.fit(all_text)

  #? Transfrom text to numerical feature
  user_bio_features = vectorizer.transform(users_df['bio']).toarray()
  repo_desc_features = vectorizer.transform(repos_df['description']).toarray()
  repo_lang_features = vectorizer.transform(repos_df['language']).toarray()

  #? Reshape star count into 2d array
  repo_stars = repos_df[['stargazers_count']].to_numpy(dtype=np.float32)
  repo_features = np.concatenate([repo_desc_features, repo_lang_features, repo_stars], axis=1)

  print(f'User feature shape: {user_bio_features.shape}')
  print(f'Repository feature shape: {repo_features.shape}')

  return user_bio_features, repo_features



# --- 3. GraphTensor Creation ---

def create_graph_tensor(users_df, repos_df, stars_df, user_features, repo_features):
  """
  Assambles the final GraphTensor object
  """
  #? Create mapping from unique IDs (login/fullname)
  user_map = {login: i for i, login in enumerate(users_df['login'])}
  repo_map = {name: i for i, name in enumerate(repos_df['fullname'])}

  #? Map the source and target of edges to integer indices
  source_indices = stars_df['source'].map(user_map).to_numpy()
  target_indices = stars_df['target'].map(repo_map).to_numpy()

  graph = tfgnn.GraphTensor.from_pieces(
    node_sets={
      'user': tfgnn.NodeSet.from_fields(
        sizes = [len(users_df)],
        features = {'feat': tf.constant(user_features, dtype=tf.float32)}
      ),
      'repo': tfgnn.NodeSet.from_fields(
        sizes = [len(repos_df)],
        features = {'feat': tf.constant(repo_features, dtype=tf.float32)}
      )
    },
    edge_sets = {
      'stars': tfgnn.EdgeSet.from_fields(
        sizes = [len(stars_df)],
        adjacency = tfgnn.Adjacency.from_indices(
          source=('user', tf.constant(source_indices, dtype=tf.int32)),
          target=('repo', tf.constant(target_indices, dtype=tf.int32))
        )
      )
    }
  )

  return graph