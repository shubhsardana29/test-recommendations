from flask import Flask, request, jsonify , render_template
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datetime import datetime
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


# Load and preprocess data
users = pd.read_csv("user.csv")
products = pd.read_csv("product.csv")
interactions = pd.read_csv("user_interaction.csv")
transactions = pd.read_csv("transactions.csv")

# Load collaborative filtering model and content similarity matrix from pickle files
with open('collaborative_model.pkl', 'rb') as f:
    collaborative_model = pickle.load(f)
with open('content_similarities.pkl', 'rb') as f:
    content_similarities = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Calculate popularity metrics
product_popularity = interactions["product_id"].value_counts()

def calculate_popularity_scores(products):
    popularity_scores = [product_popularity.get(product_id, 0) for product_id in products["product_id"]]
    normalized_scores = popularity_scores / np.max(popularity_scores)
    return normalized_scores

# Calculate user similarity using Jaccard similarity
def calculate_user_similarity_scores(user_id):
    user1_interactions = set(interactions[interactions["user_id"] == user_id]["product_id"])
    similarity_scores = []
    
    for other_user_id in interactions["user_id"].unique():
        if other_user_id != user_id:
            user2_interactions = set(interactions[interactions["user_id"] == other_user_id]["product_id"])
            common_interactions = user1_interactions.intersection(user2_interactions)
            jaccard_similarity = len(common_interactions) / len(user1_interactions.union(user2_interactions))
            similarity_scores.append((other_user_id, jaccard_similarity))
    
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return similarity_scores

# Calculate recency scores
def calculate_recency_scores(user_id):
    user_interactions = interactions[interactions["user_id"] == user_id]
    current_time = datetime.now()
    recency_scores = [(current_time - pd.to_datetime(ts)).days for ts in user_interactions["timestamp"]]
    
    print("Recency scores:", recency_scores)  # Debug: Print the recency scores

    if not recency_scores:
        return [0]  # Return a default value if the list is empty
    
    max_recency = max(recency_scores)
    normalized_scores = [1 - (score / max_recency) for score in recency_scores]  # Normalize to values between 0 and 1
    
    print("Normalized scores:", normalized_scores)  # Debug: Print the normalized scores
    
    return normalized_scores

# Combine scores using weighted averages
def personalized_rankings(user_id, num_rankings=10):
    user_rated_products = interactions[interactions["user_id"] == user_id]["product_id"].tolist()
    
    # Calculate recency scores once for the user
    recency_scores = calculate_recency_scores(user_id)

    testset = []
    for product_id in products["product_id"].tolist():
        if product_id not in user_rated_products:
            testset.append((user_id, product_id, 0))
    
    collaborative_scores = []
    content_scores = []
    popularity_scores = []
    user_similarity_scores = []
    recency_scores_for_products = []

    
    for uid, iid, _ in testset:
        collaborative_scores.append(collaborative_model.predict(uid, iid).est)
    
        if iid in user_rated_products:
            content_scores.append(content_similarities[products[products["product_id"] == iid].index[0]])
        else:
            content_scores.append(0)  # Use a default score if the product doesn't have content score
        
        popularity_scores.append(calculate_popularity_scores(products)[products["product_id"] == iid][0])
        user_similarity_scores.append(calculate_user_similarity_scores(user_id)[0][1])

        recency_score_index = user_rated_products.index(iid) if iid in user_rated_products else -1
        recency_scores_for_product = recency_scores[recency_score_index] if recency_score_index >= 0 else 0
        recency_scores_for_products.append(recency_scores_for_product)

    num_products = len(testset)
    num_collaborative = len(collaborative_scores)
    num_content = len(content_scores)
    num_popularity = len(popularity_scores)
    num_similarity = len(user_similarity_scores)
    num_recency = len(recency_scores_for_products)

    print("Num products:", num_products)
    print("Num collaborative:", num_collaborative)
    print("Num content:", num_content)
    print("Num popularity:", num_popularity)
    print("Num similarity:", num_similarity)
    print("Num recency:", num_recency)

    assert (
        num_products == num_collaborative == num_content == num_popularity == num_similarity == num_recency
    ), "All arrays must have the same length"
  

    recency_scores_for_products = [recency_scores_for_product] * len(testset)

    # Create a DataFrame to hold the scores
    scores_df = pd.DataFrame({
        "product_id": [iid for _, iid, _ in testset],
        "collaborative": collaborative_scores,
        "content": content_scores,
        "popularity": popularity_scores,
        "user_similarity": user_similarity_scores,
        "recency": recency_scores_for_products
    })
    
    # Define the weights for combining scores
    alpha = 0.3
    beta = 0.3
    gamma = 0.2
    delta = 0.1
    epsilon = 0.1

    # Fill missing values with 0
    scores_df = scores_df.fillna(0)

    # Combine scores using weighted averages
    scores_df["combined_score"] = (
        alpha * scores_df["collaborative"] +
        beta * scores_df["content"] +
        gamma * scores_df["popularity"] +
        delta * scores_df["user_similarity"] +
        epsilon * scores_df["recency"]
    )
    
    # Rank products based on combined scores
    ranked_products = scores_df.sort_values(by="combined_score", ascending=False).head(num_rankings)
    
    return ranked_products

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_id = request.json['user_id']
    num_rankings = request.json.get('num_rankings', 10)

    # Calculate personalized rankings for the user
    rankings = personalized_rankings(user_id, num_rankings)

    # Convert the rankings to a JSON response
    response = {
        'user_id': user_id,
        'rankings': rankings.to_dict(orient='records')
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)