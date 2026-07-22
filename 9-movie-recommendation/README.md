# Movie Recommendation System (SVD-based)

A collaborative filtering movie recommendation system built using Singular Value 
Decomposition (SVD), trained on the MovieLens dataset. Given a user ID, the model 
predicts and ranks movies the user is likely to enjoy.

## 📌 Overview

This project implements a recommendation engine that learns latent patterns in 
user-movie ratings using matrix factorization (SVD). Given an existing user ID, 
it outputs a ranked list of movies with predicted ratings.

## 🚀 Features

- SVD-based collaborative filtering for personalized recommendations
- Trained on the MovieLens `ml-latest-small` dataset
- Interactive prediction — enter a User ID and get top movie recommendations
- Outputs movie titles merged with predicted ratings for readability

## 🗂️ Project Structure

```
├── Movies Recommendation System.ipynb   # Main notebook (data prep, training, predictions)
├── ml-latest-small/                     # MovieLens dataset (ratings, movies, etc.)
└── README.md
```

## 🛠️ Tech Stack

- Python
- Jupyter Notebook
- pandas, numpy
- scikit-learn / scipy (SVD / matrix factorization)

## ⚙️ Setup & Installation

1. Clone the repository:
```bash
   git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
   cd movie-recommendation-system
```

2. Install dependencies:
```bash
   pip install jupyter pandas numpy scikit-learn scipy
```

3. Launch Jupyter:
```bash
   jupyter notebook
```

4. Open `Movies Recommendation System.ipynb` and run all cells 
   (**Kernel → Restart & Run All**).

## ▶️ Usage

When prompted, enter a valid **User ID** from the dataset (an integer, e.g. `1`, `42`):

```
Enter User ID: 1
```

The notebook will output a table of recommended movies:

| movieId | title                  | Predicted Rating |
|---------|------------------------|-------------------|
| ...     | ...                    | ...               |

To check valid user IDs available in the dataset:
```python
import pandas as pd
ratings = pd.read_csv("ml-latest-small/ratings.csv")
print(ratings["userId"].unique())
```

## 📊 Dataset

This project uses the **MovieLens `ml-latest-small`** dataset, which contains 
user ratings for movies along with movie metadata (titles, genres).

## 📄 License

This project is open source and available under the MIT License.