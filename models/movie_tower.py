import torch
import torch.nn as nn

class MovieTower(nn.Module):
    def __init__(self, num_movies, movie_feature_dim, embedding_dim=32):
        super(MovieTower, self).__init__()

        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(movie_feature_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, movie_id, movie_features):
        movie_embed = self.movie_embedding(movie_id)
        x = torch.cat([movie_embed, movie_features], dim=1)
        return self.fc(x)
