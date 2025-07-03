import torch
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'stars', 'repo'): SAGEConv((-1, -1), hidden_channels),
            ('repo', 'rev_stars', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('user', 'stars', 'repo'): SAGEConv((-1, -1), out_channels),
            ('repo', 'rev_stars', 'user'): SAGEConv((-1, -1), out_channels),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        hidden_embeds = self.conv1(x_dict, edge_index_dict)
        hidden_embeds = {key: x.relu() for key, x in hidden_embeds.items()}
        final_embeds = self.conv2(hidden_embeds, edge_index_dict)
        return final_embeds

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
      super().__init__()
      self.encoder = HeteroGNN(hidden_channels, out_channels)
    
    def forward(self, data):
      embeddings = self.encoder(data.x_dict, data.edge_index_dict)
      return embeddings



print("--- Loading model and assets... ---")

ASSETS_PATH = "./model_assets" 
MODEL_STATE_PATH = f"{ASSETS_PATH}/gnn_model_state.pth"
USER_MAP_PATH = f"{ASSETS_PATH}/user_map.pkl"
REPO_MAP_PATH = f"{ASSETS_PATH}/repo_map.pkl"

with open(USER_MAP_PATH, 'rb') as f:
  user_map = pickle.load(f)
with open(REPO_MAP_PATH, 'rb') as f:
  repo_map = pickle.load(f)

#? Invert maps for easy lookup from index to name
idx_to_repo = {i: name for name, i in repo_map.items()}


#* --- Load the trained model ---
trained_model = Model(hidden_channels=64, out_channels=32)
trained_model.load_state_dict(torch.load(MODEL_STATE_PATH))
trained_model.eval()

from data_loader import create_hetero_data_object
import torch_geometric.transforms as T

DATA_PATH = './exported_data'
users_df = pd.read_csv(f"{DATA_PATH}/users.csv")
repos_df = pd.read_csv(f"{DATA_PATH}/repos.csv")
stars_df = pd.read_csv(f"{DATA_PATH}/stars.csv")

num_users = len(users_df)
num_repos = len(repos_df)

#! harus sama dengan model yg di train
user_feature_dim = 128
repo_feature_dim = 257
dummy_user_features = torch.randn(num_users, user_feature_dim)
dummy_repo_features = torch.randn(num_repos, repo_feature_dim)

graph_data = create_hetero_data_object(users_df, repos_df, stars_df, dummy_user_features, dummy_repo_features)
graph_data_undirected = T.ToUndirected()(graph_data)


#* ---- Generate All Embeddings Once at Startup ---
print("--- Generating all node embeddings... ---")
with torch.no_grad():
  all_embeddings = trained_model(graph_data_undirected)
  user_embeddings = all_embeddings['user']
  repo_embeddings = all_embeddings['repo']
print("--- Embeddings generated. Server is ready! ---")


#* ---- FastAPI ---

app = FastAPI(title = "GitGraph API")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

@app.get('/recommend/{username}', tags=["Recommendation"])
def get_recommendations(username: str, top_k: int = 10):
  """
  Ngasih rekomendasi dari username github yg dikasih
  """

  print(f'Received recommendation request for user: {username}')
  if username not in user_map:
    raise HTTPException(status_code=404, detail=f"User '{username}' not found in the dataset")
  
  user_idx = user_map[username]
  target_user_embedding = user_embeddings[user_idx]

  similarities = torch.nn.functional.cosine_similarity(
    target_user_embedding.unsqueeze(0),
    repo_embeddings
  )

  top_k_indices = torch.topk(similarities, k=top_k).indices
  recommended_repos = [idx_to_repo[idx.item()] for idx in top_k_indices]

  return {
    "user": username,
    "recommendations": recommended_repos,
    "similarity_scores": torch.topk(similarities, k=top_k).values.tolist()
  }