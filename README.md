# Two-Tower Recommendation System for Netflix Movies

A deep learning-based recommendation system leveraging a Two-Tower architecture to predict user ratings and generate personalized movie recommendations. Built using PyTorch, this model trains on the Netflix Prize Dataset to learn meaningful user and movie embeddings.

## Overview

This project implements a Two-Tower neural network for movie recommendations. The model learns separate representations for users and movies and predicts ratings based on their embeddings.

## Architecture

The recommendation model consists of two main components:

### Two-Tower Model

- **User Tower:**
  - Encodes users using learned embeddings and additional user features.
  - Passes the user representation through fully connected layers with ReLU activations.

- **Movie Tower:**
  - Encodes movies using learned embeddings and metadata (e.g., release year).
  - Processes the embeddings through a similar fully connected network as the User Tower.

- **Dot-Product Similarity:**
  - Computes similarity between user and movie embeddings using a dot product.
  - Applies a sigmoid activation to output a predicted rating between 1 and 5 stars.

## Model Flow

1. User and movie data are embedded and processed separately through independent networks.
2. User and movie embeddings are projected into the same latent space, capturing relationships between them.
3. A dot-product operation measures the similarity between the two embeddings.
4. The final score is mapped to a predicted rating using a sigmoid function scaled to the [1, 5] range.
5. The loss function (Here, Mean Squared Error) optimizes the model to minimize the difference between predicted and actual ratings.


## Training Process

1. **Input:** The model is trained on user-movie interactions (ratings from the Netflix dataset).
2. **Optimization:** The objective is to minimize a loss function (MSE) that compares predicted ratings to actual user ratings.
3. **Backpropagation:** The network learns embeddings for users and movies based on their interactions.


## Inference (Making Recommendations)

- The trained model generates predicted ratings for all unseen movies for a given user.
- The top N movies with the highest predicted ratings are recommended.

## Dataset

This model is trained on the Netflix Prize Dataset:  
[Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data?select=probe.txt)

### Dataset Files

- `combined_data_1.txt` to `combined_data_4.txt` → Contains user ratings per movie (UserID, Rating, Date).
- `movie_titles.csv` → Metadata about movies (MovieID, Year of Release, Title).


## Usage

### Preprocess the Data

```bash
python preprocessing/netflix_preprocess.py
```

### Train the Model

```bash
python main.py --epochs 100 --batch_size 128 --lr 0.001
```

### Generate Recommendations

```bash
python rec.py
```
