import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import os

PROCESSED_DIR = os.path.join("data", "processed")
RATINGS_PATH = os.path.join(PROCESSED_DIR, "ratings_processed.csv")
USERS_PATH = os.path.join(PROCESSED_DIR, "users_processed.csv")
MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_processed.csv")

def load_processed_data():
    ratings_df = pd.read_csv(RATINGS_PATH)
    users_df = pd.read_csv(USERS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)

    # ratings_df = ratings_df.sample(frac=0.01, random_state=42) # To prototype/verify pipeline
    
    ratings_df["user_id"] = ratings_df["user_id"].astype("category")
    ratings_df["movie_id"] = ratings_df["movie_id"].astype("category")
    users_df["user_id"] = users_df["user_id"].astype("category")
    movies_df["movie_id"] = movies_df["movie_id"].astype("category")
    
    user_id_map = {uid: idx for idx, uid in enumerate(users_df["user_id"].cat.categories)}
    movie_id_map = {mid: idx for idx, mid in enumerate(movies_df["movie_id"].cat.categories)}
    
    ratings_df["user_id"] = ratings_df["user_id"].map(user_id_map)
    ratings_df["movie_id"] = ratings_df["movie_id"].map(movie_id_map)
    users_df["user_id"] = users_df["user_id"].map(user_id_map)
    movies_df["movie_id"] = movies_df["movie_id"].map(movie_id_map)
    
    ratings_df = ratings_df.dropna(subset=["user_id", "movie_id"])
    ratings_df["user_id"] = ratings_df["user_id"].astype(int)
    ratings_df["movie_id"] = ratings_df["movie_id"].astype(int)
    
    if "dummy_feature" not in users_df.columns:
        users_df["dummy_feature"] = 1.0
    global_user_features = users_df.set_index("user_id").sort_index()[["dummy_feature"]].fillna(0)
    
    min_year = movies_df["year_of_release"].min()
    max_year = movies_df["year_of_release"].max()
    movies_df["year_of_release_norm"] = (movies_df["year_of_release"] - min_year) / (max_year - min_year)
    global_movie_features = movies_df.set_index("movie_id").sort_index()[["year_of_release_norm"]].fillna(0)
    
    user_features_per_rating = torch.tensor(global_user_features.loc[ratings_df["user_id"]].values, dtype=torch.float32)
    movie_features_per_rating = torch.tensor(global_movie_features.loc[ratings_df["movie_id"]].values, dtype=torch.float32)
    
    ratings = torch.tensor(ratings_df["rating"].values, dtype=torch.float32)
    user_ids = torch.tensor(ratings_df["user_id"].values, dtype=torch.long)
    movie_ids = torch.tensor(ratings_df["movie_id"].values, dtype=torch.long)
      
    return user_ids, user_features_per_rating, movie_ids, movie_features_per_rating, ratings

def get_data_loaders(batch_size=64, train_ratio=0.8):
    user_ids, user_features, movie_ids, movie_features, ratings = load_processed_data()
    dataset = TensorDataset(user_ids, user_features, movie_ids, movie_features, ratings)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
