import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, num_users, user_feature_dim, embedding_dim=32):
        super(UserTower, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(user_feature_dim + embedding_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, user_id, user_features):
        user_embed = self.user_embedding(user_id)
        x = torch.cat([user_embed, user_features], dim=1) 
        return self.fc(x)
