import torch
import torch.nn as nn
from models.user_tower import UserTower
from models.movie_tower import MovieTower

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, user_feature_dim, movie_feature_dim, embedding_dim=32):
        super(TwoTowerModel, self).__init__()

        self.user_tower = UserTower(num_users, user_feature_dim, embedding_dim)
        self.movie_tower = MovieTower(num_movies, movie_feature_dim, embedding_dim)

    def forward(self, user_id, user_features, movie_id, movie_features):
        user_embedding = self.user_tower(user_id, user_features)
        movie_embedding = self.movie_tower(movie_id, movie_features)

        similarity = torch.sum(user_embedding * movie_embedding, dim=1)
        return similarity
