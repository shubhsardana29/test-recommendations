import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV files
users_data = pd.read_csv("user.csv")
products_data = pd.read_csv("products.csv")
categories_data = pd.read_csv("categories.csv")
brands_data = pd.read_csv("brands.csv")
user_interaction_data = pd.read_csv("user_interaction.csv")
transactions_data = pd.read_csv("transactions.csv")

# Preprocessing
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()
category_encoder = LabelEncoder()
brand_encoder = LabelEncoder()

users_data["user_id"] = user_encoder.fit_transform(users_data["user_id"])
products_data["product_id"] = product_encoder.fit_transform(products_data["product_id"])
categories_data["category_id"] = category_encoder.fit_transform(categories_data["category_id"])
brands_data["brand_id"] = brand_encoder.fit_transform(brands_data["brand_id"])

# Convert category and brand columns to strings
categories_data["category_id"] = categories_data["category_id"].astype(str)
brands_data["brand_id"] = brands_data["brand_id"].astype(str)

# Merge product data with category and brand
products_data = pd.merge(products_data, categories_data, left_on="category", right_on="category_id", how="left")
products_data = pd.merge(products_data, brands_data, left_on="brand", right_on="brand_id", how="left")

# Merge interaction data with transaction data
interactions_data = pd.concat([user_interaction_data, transactions_data], ignore_index=True)
interactions_data["interaction_type"] = interactions_data["interaction_type"].replace("purchase", 2)
interactions_data["interaction_type"] = interactions_data["interaction_type"].replace("add_to_cart", 1)
interactions_data["interaction_type"] = interactions_data["interaction_type"].replace("view", 0)

# Split data into train and test sets
train_data, test_data = train_test_split(interactions_data, test_size=0.2, random_state=42)


def collaborative_filtering_recommendations(user_id, num_recommendations):
    user_interactions = interactions_data[interactions_data["user_id"] == user_id]
    
    # Aggregate duplicate entries by taking the maximum interaction_type value
    user_interactions = user_interactions.groupby(["user_id", "product_id"])["interaction_type"].max().reset_index()
    
    user_product_matrix = user_interactions.pivot(index="user_id", columns="product_id", values="interaction_type").fillna(0)
    
    nmf_model = NMF(n_components=10, random_state=42)
    nmf_model.fit(user_product_matrix)
    
    user_profile = nmf_model.transform(user_product_matrix.loc[user_id].values.reshape(1, -1))
    predicted_scores = np.dot(user_profile, nmf_model.components_).flatten()
    
    recommended_product_indices = np.argsort(predicted_scores)[::-1][:num_recommendations]
    recommended_products = [product_encoder.classes_[i] for i in recommended_product_indices]
    
    return recommended_products


def content_based_recommendations(user_id, num_recommendations):
    user_interactions = interactions_data[interactions_data["user_id"] == user_id]
    user_interacted_products = user_interactions["product_id"].tolist()

    user_preferences = products_data[products_data["product_id"].isin(user_interacted_products)]
    user_preferences = user_preferences[["category_id", "brand_id"]]  # Simplified for illustration

    product_profiles = products_data[["category_id", "brand_id"]]  # Simplified for illustration

    # Drop rows with missing values
    user_preferences = user_preferences.dropna()
    product_profiles = product_profiles.dropna()

    if user_preferences.empty or product_profiles.empty:
        return []  # Return an empty list if there are no valid preferences or profiles

    # Reshape user_preferences and product_profiles for cosine_similarity
    user_preferences_reshaped = user_preferences.values.reshape(1, -1)
    product_profiles_reshaped = product_profiles.values

    cosine_sim = cosine_similarity(user_preferences_reshaped, product_profiles_reshaped).flatten()
    recommended_product_indices = np.argsort(cosine_sim)[::-1][:num_recommendations]
    recommended_products = [product_encoder.classes_[i] for i in recommended_product_indices]

    return recommended_products


def find_similar_users(user_id):
    # Placeholder function to find similar users based on user interactions
    # You need to implement your own logic to calculate user similarity
    
    # For demonstration purposes, let's assume user similarity is based on the same products they interacted with
    user_interactions = interactions_data[interactions_data["user_id"] == user_id]
    user_interacted_products = user_interactions["product_id"].tolist()
    
    # Find users who interacted with the same products
    similar_users = interactions_data[interactions_data["product_id"].isin(user_interacted_products)]
    similar_users = similar_users[similar_users["user_id"] != user_id]
    similar_user_ids = similar_users["user_id"].unique().tolist()
    
    return similar_user_ids

def user_similarity_based_recommendations(user_id, num_recommendations):
    user_interactions = interactions_data[interactions_data["user_id"] == user_id]
    user_interacted_products = user_interactions["product_id"].tolist()

    similar_users = find_similar_users(user_id)

    # Calculate the similarity score between the user and each similar user
    similarity_scores = []
    for other_user_id in similar_users:
        other_user_interactions = interactions_data[interactions_data["user_id"] == other_user_id]
        other_user_interacted_products = other_user_interactions["product_id"].tolist()
        
        # Calculate Jaccard similarity between user interactions
        similarity = len(set(user_interacted_products).intersection(other_user_interacted_products)) / len(set(user_interacted_products).union(other_user_interacted_products))
        similarity_scores.append((other_user_id, similarity))
    
    # Sort similar users by similarity score in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get top similar users and their interactions
    top_similar_users = similarity_scores[:num_recommendations]
    recommended_products = []
    for other_user_id, _ in top_similar_users:
        other_user_interactions = interactions_data[interactions_data["user_id"] == other_user_id]
        recommended_products.extend(other_user_interactions["product_id"].tolist())
    
    # Remove duplicates and products already interacted by the user
    recommended_products = list(set(recommended_products) - set(user_interacted_products))

    return recommended_products

def popularity_based_recommendations(num_recommendations):
    popular_products = interactions_data.groupby("product_id")["interaction_type"].sum()
    popular_products = popular_products.sort_values(ascending=False).index[:num_recommendations]

    return popular_products.tolist()


def generate_hybrid_recommendations(user_id, num_recommendations):
    cf_recommendations = collaborative_filtering_recommendations(user_id, num_recommendations)
    cb_recommendations = content_based_recommendations(user_id, num_recommendations)
    user_similarity_recommendations = user_similarity_based_recommendations(user_id, num_recommendations)
    popularity_recommendations = popularity_based_recommendations(num_recommendations)

    # Combine recommendations from different methods and remove duplicates
    all_recommendations = cf_recommendations + cb_recommendations + user_similarity_recommendations + popularity_recommendations

    # Calculate scores for unique recommendations based on their occurrence
    recommendation_scores = {}
    for product_id in all_recommendations:
        if product_id not in recommendation_scores:
            recommendation_scores[product_id] = 0
        recommendation_scores[product_id] += 1

    # Sort recommendations by scores in descending order
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

   # Extract product IDs and scores from sorted recommendations
    final_recommendations = [(product_id, score) for product_id, score in sorted_recommendations]

    return final_recommendations[:num_recommendations]

# Assuming you have loaded interactions_data, products_data, and other necessary data

# # Get user input
# user_id = int(input("Enter user ID: "))
# num_recommendations = int(input("Enter number of recommendations: "))

# hybrid_recommendations = generate_hybrid_recommendations(user_id, num_recommendations)

# print("Hybrid Recommendations for user_id=",user_id)
# for product_id, score in hybrid_recommendations:
#     print(f"Product ID: {product_id} | Score: {score}")
