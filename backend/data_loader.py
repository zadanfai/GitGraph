import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocess_data(users_df, repos_df, vectorizer):
  users_df['bio'].fillna('', inplace=True)
  repos_df['description'].fillna('', inplace=True)
  repos_df['language'].fillna('', inplace=True)
  repos_df['stargazers_count'].fillna(0, inplace=True)

  user_bio_features = vectorizer.transform(users_df['bio']).toarray()
  repo_desc_features = vectorizer.transform(repos_df['description']).toarray()
  repo_lang_features = vectorizer.transform(repos_df['language']).toarray()

  repo_stars = repos_df[['stargazers_count']].to_numpy(dtype=np.float32)
  repo_features = np.concat([repo_desc_features, repo_lang_features, repo_stars], axis=1)

  return user_bio_features, repo_features


def create_hetero_data_object(users_df, repos_df, stars_df, user_features, repo_features, ):

  data = HeteroData()

  data['user'].x = torch.tensor(user_features, dtype=torch.float)
  data['repo'].x = torch.tensor(repo_features, dtype=torch.float)

  user_map = {login: i for i, login in enumerate(users_df['login'])}
  repo_map = {name: i for i, name in enumerate(repos_df['full_name'])}

  source_indices = torch.tensor([user_map.get(login) for login in stars_df['source']], dtype=torch.long)
  target_indices = torch.tensor([repo_map.get(name) for name in stars_df['target']], dtype=torch.long)

  edge_index = torch.stack([source_indices, target_indices], dim=0)

  data['user', 'stars', 'repo'].edge_index = edge_index

  return data