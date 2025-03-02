import os
import pandas as pd
import torch
from models.two_tower import TwoTowerModel

def build_mappings():
    MOVIES_PROCESSED = os.path.join("data", "processed", "movies_processed.csv")
    movies_df = pd.read_csv(MOVIES_PROCESSED)
    movies_df["movie_id"] = movies_df["movie_id"].astype("category")
    categories = movies_df["movie_id"].cat.categories
    code_to_original_movie = {code: categories[code] for code in range(len(categories))}
    movieid_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))
    return code_to_original_movie, movieid_to_title

def load_global_features():
    PROCESSED_DIR = os.path.join("data", "processed")
    USERS_PATH = os.path.join(PROCESSED_DIR, "users_processed.csv")
    MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_processed.csv")
    users_df = pd.read_csv(USERS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)
    
    if "dummy_feature" not in users_df.columns:
        users_df["dummy_feature"] = 1.0
    
    users_df["user_id"] = users_df["user_id"].astype("category").cat.codes
    movies_df["movie_id"] = movies_df["movie_id"].astype("category").cat.codes
    
    global_user_features = users_df.set_index("user_id").sort_index()[["dummy_feature"]].fillna(0)
    feature_col = "year_of_release_norm" if "year_of_release_norm" in movies_df.columns else "year_of_release"
    global_movie_features = movies_df.set_index("movie_id").sort_index()[[feature_col]].fillna(0)
    
    user_features_tensor = torch.tensor(global_user_features.values, dtype=torch.float32)
    movie_features_tensor = torch.tensor(global_movie_features.values, dtype=torch.float32)
    return users_df, movies_df, user_features_tensor, movie_features_tensor

def get_user_interactions(user_id):
    PROCESSED_DIR = os.path.join("data", "processed")
    RATINGS_PATH = os.path.join(PROCESSED_DIR, "ratings_processed.csv")
    MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_processed.csv")
    ratings_df = pd.read_csv(RATINGS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)
    
    movies_df["movie_id"] = movies_df["movie_id"].astype("category")
    categories = movies_df["movie_id"].cat.categories
    ratings_df["movie_id"] = pd.Categorical(ratings_df["movie_id"], categories=categories)
    ratings_df["movie_id"] = ratings_df["movie_id"].cat.codes
    ratings_df["user_id"] = ratings_df["user_id"].astype(int)
    
    interacted = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"].unique())
    return interacted

def recommend_for_user(model, user_id, user_features, movie_features, interacted_movies, top_n=5, device="cpu"):
    model.eval()
    user_feat = user_features[user_id].unsqueeze(0).to(device)
    user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
    with torch.no_grad():
        raw_preds = model.user_tower(user_id_tensor, user_feat)
        all_movie_ids = torch.arange(movie_features.shape[0], dtype=torch.long).to(device)
        all_movie_feats = movie_features.to(device)
        movie_embeddings = model.movie_tower(all_movie_ids, all_movie_feats)
        raw_scores = torch.matmul(movie_embeddings, raw_preds.squeeze(0))
        preds = torch.sigmoid(raw_scores) * 4 + 1
        preds_np = preds.cpu().numpy()
        
        sorted_indices = preds_np.argsort()[::-1]
        filtered_indices = [idx for idx in sorted_indices if idx not in interacted_movies]
        recommended_indices = filtered_indices[:top_n]
        recommended_scores = preds_np[recommended_indices]
    return recommended_indices, recommended_scores

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code_to_original_movie, movieid_to_title = build_mappings()
    users_df, movies_df, user_features, movie_features = load_global_features()
    num_users = users_df["user_id"].nunique()
    num_movies = movies_df["movie_id"].nunique()
    print(f"Loaded {num_users} users and {num_movies} movies.")
    
    user_feature_dim = user_features.shape[1]
    movie_feature_dim = movie_features.shape[1]
    embedding_dim = 32
    
    model = TwoTowerModel(num_users, num_movies, user_feature_dim, movie_feature_dim, embedding_dim)
    model.load_state_dict(torch.load("two_tower_model.pth", map_location=device))
    model.to(device)
    
    example_user_id = 10
    print(f"Generating recommendations for user {example_user_id}...")
    interacted_movies = get_user_interactions(example_user_id)
    print(f"User {example_user_id} has interacted with movie codes: {interacted_movies}")
    
    print("Movies the user has seen:")
    for code in sorted(interacted_movies):
        orig_movie = code_to_original_movie.get(code, None)
        if orig_movie is not None:
            title = movieid_to_title.get(orig_movie, f"Movie {orig_movie}")
            print(f"  - code {code} => Movie ID '{orig_movie}', Title: {title}")
    
    recommended_ids, scores = recommend_for_user(model, example_user_id, user_features, movie_features, interacted_movies, top_n=5, device=device)
    
    print("Top recommendations:")
    for movie_id, score in zip(recommended_ids, scores):
        orig_movie = code_to_original_movie.get(movie_id, None)
        if orig_movie is not None:
            title = movieid_to_title.get(orig_movie, f"Movie {orig_movie}")
            print(f"  - Movie code {movie_id}, Score {score:.4f}")
            print(f"    => Movie ID '{orig_movie}', Title: '{title}'")

if __name__ == "__main__":
    main()
