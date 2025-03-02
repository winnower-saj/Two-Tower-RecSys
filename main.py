import pandas as pd
import torch
import argparse
from utils.data_loader import get_data_loaders
from models.two_tower import TwoTowerModel
from utils.train_eval import train, evaluate
import os

PROCESSED_DIR = os.path.join("data", "processed")
USERS_PATH = os.path.join(PROCESSED_DIR, "users_processed.csv")
MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_processed.csv")

def get_dataset_stats():
    users_df = pd.read_csv(USERS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)
    users_df["user_id"] = users_df["user_id"].astype("category").cat.codes
    movies_df["movie_id"] = movies_df["movie_id"].astype("category").cat.codes
    num_users = users_df["user_id"].nunique()
    num_movies = movies_df["movie_id"].nunique()
    return num_users, num_movies

def parse_args():
    parser = argparse.ArgumentParser(description="Train Two-Tower Model on Netflix Data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    num_users, num_movies = get_dataset_stats()
    print(f"Detected {num_users} users and {num_movies} movies")
    
    user_feature_dim = 1
    movie_feature_dim = 1
    embedding_dim = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TwoTowerModel(num_users, num_movies, user_feature_dim, movie_feature_dim, embedding_dim).to(device)
    print("Starting training...")
    train(model, train_loader, epochs=args.epochs, lr=args.lr)
    print("Evaluating model...")
    evaluate(model, test_loader)
    torch.save(model.state_dict(), "two_tower_model.pth")

if __name__ == "__main__":
    main()
