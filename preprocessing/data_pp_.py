import os
import glob
import pandas as pd
import concurrent.futures

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

NETFLIX_FILE_PATTERN = os.path.join(DATA_DIR, "combined_data_*.txt")
MOVIE_TITLES_FILE = os.path.join(DATA_DIR, "movie_titles.csv")

PROCESSED_RATING_FILE = os.path.join(PROCESSED_DIR, "ratings_processed.csv")
PROCESSED_MOVIE_FILE = os.path.join(PROCESSED_DIR, "movies_processed.csv")
PROCESSED_USER_FILE = os.path.join(PROCESSED_DIR, "users_processed.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def parse_file(file_path):
    rating_records = []
    current_movie_id = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                try:
                    user_id_str, rating_str, date_str = line.split(',')
                    rating_records.append({
                        'movie_id': current_movie_id,
                        'user_id': int(user_id_str),
                        'rating': int(rating_str),
                        'date': date_str
                    })
                except ValueError:
                    continue
    return rating_records

def main():
    PROTOTYPE = True # to quickly verify pipeline, set to true (essentially considers a smaller sample of data)

    print("Processing Netflix data{}...".format(" (prototype subset)" if PROTOTYPE else ""))
    
    file_list = glob.glob(NETFLIX_FILE_PATTERN)
    if PROTOTYPE:
        max_files = 1 
        file_list = file_list[:max_files]
    print(f"Found {len(file_list)} file(s) to process{' (subset)' if PROTOTYPE else ''}.")

    all_ratings = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(parse_file, file_list)
        for rating_list in results:
            all_ratings.extend(rating_list)
    
    ratings_df = pd.DataFrame(all_ratings)
    print(f"Total rating rows parsed: {len(ratings_df):,}")
    
    ratings_df.to_csv(PROCESSED_RATING_FILE, index=False)
    print(f"Ratings saved to {PROCESSED_RATING_FILE}")

    print("Processing movie_titles.csv...")
    movies_df = pd.read_csv(
        MOVIE_TITLES_FILE, 
        header=None, 
        names=["movie_id", "year_of_release", "title"],
        encoding="latin-1",
        on_bad_lines='skip',
        engine="python"
    )
    
    unique_movies = ratings_df["movie_id"].unique()
    movies_df = movies_df[movies_df["movie_id"].isin(unique_movies)].copy()
    
    movies_df["year_of_release"] = pd.to_numeric(movies_df["year_of_release"], errors="coerce").fillna(0).astype(int)
    
    movies_df.to_csv(PROCESSED_MOVIE_FILE, index=False)
    print(f"Movies saved to {PROCESSED_MOVIE_FILE} with {len(movies_df):,} rows")
    
    print("Generating users file...")
    unique_users = ratings_df["user_id"].unique()
    users_df = pd.DataFrame({"user_id": unique_users})
    
    
    users_df.to_csv(PROCESSED_USER_FILE, index=False)
    print(f"Users saved to {PROCESSED_USER_FILE} with {len(users_df):,} rows")
    

if __name__ == '__main__':
    main()
