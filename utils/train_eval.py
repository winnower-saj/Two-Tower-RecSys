import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np

def train(model, train_loader, epochs=5, lr=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for user_ids, user_features, movie_ids, movie_features, ratings in train_loader:
            user_ids = user_ids.to(device)
            user_features = user_features.to(device)
            movie_ids = movie_ids.to(device)
            movie_features = movie_features.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            raw_preds = model(user_ids, user_features, movie_ids, movie_features).squeeze()
            preds = torch.sigmoid(raw_preds) * 4 + 1
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")

def evaluate(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for user_ids, user_features, movie_ids, movie_features, labels in test_loader:
            user_ids = user_ids.to(device)
            user_features = user_features.to(device)
            movie_ids = movie_ids.to(device)
            movie_features = movie_features.to(device)
            labels = labels.to(device)

            raw_preds = model(user_ids, user_features, movie_ids, movie_features).squeeze()
            preds = torch.sigmoid(raw_preds) * 4 + 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    mse = np.mean((all_labels - all_preds) ** 2)
    rmse = np.sqrt(mse)
    print(f"Evaluation RMSE: {rmse:.4f}")
    return rmse
