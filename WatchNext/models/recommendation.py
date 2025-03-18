import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import random


class RecommendationEngine:
    """
    Simple recommendation engine that demonstrates both collaborative and content-based filtering
    techniques but using a small dataset for demonstration purposes.
    """

    def __init__(self, movies_path='data/movies.json', users_path='data/users.json'):
        """Initialize the recommendation engine with movie and user data"""
        # Load data
        with open(movies_path, 'r') as f:
            self.movies = json.load(f)

        with open(users_path, 'r') as f:
            self.users = json.load(f)

        # Create movie features for content-based filtering
        self._prepare_content_features()

        # Create user-item matrix for collaborative filtering
        self._prepare_collaborative_features()

    def _prepare_content_features(self):
        """Prepare content features for movies using genres and descriptions"""
        # For demo purposes, we'll create a simple feature string from genres and overview
        for movie in self.movies:
            movie['feature_string'] = ' '.join(movie['genres']) + ' ' + movie.get('overview', '')

        # Create a TF-IDF matrix for movie features
        self.tfidf = TfidfVectorizer(stop_words='english')
        try:
            feature_strings = [movie['feature_string'] for movie in self.movies]
            self.tfidf_matrix = self.tfidf.fit_transform(feature_strings)
            self.content_similarity = cosine_similarity(self.tfidf_matrix)
        except:
            # Fallback if there's an issue with TF-IDF
            self.content_similarity = np.eye(len(self.movies))  # Identity matrix as fallback

    def _prepare_collaborative_features(self):
        """Prepare collaborative filtering features using user ratings"""
        # For demo purposes, we'll create a simple user-item matrix
        movie_ids = [movie['id'] for movie in self.movies]
        user_ids = list(self.users.keys())

        # Create a user-item matrix (users × movies)
        self.user_item_matrix = np.zeros((len(user_ids), len(movie_ids)))

        # Fill in ratings where available
        for i, user_id in enumerate(user_ids):
            user = self.users[user_id]
            if 'ratings' in user:
                for movie_id, rating in user['ratings'].items():
                    if movie_id in movie_ids:
                        j = movie_ids.index(movie_id)
                        self.user_item_matrix[i, j] = rating

        # Calculate user similarity using cosine similarity
        try:
            self.user_similarity = cosine_similarity(self.user_item_matrix)
        except:
            # Fallback
            self.user_similarity = np.eye(len(user_ids))  # Identity matrix as fallback

        # Store indices for quick lookup
        self.movie_indices = {movie['id']: i for i, movie in enumerate(self.movies)}
        self.user_indices = {user_id: i for i, user_id in enumerate(user_ids)}

    def recommend_content_based(self, movie_id, num_recommendations=6):
        """Content-based recommendation based on movie similarity"""
        # If movie not found, return random recommendations
        if movie_id not in self.movie_indices:
            return random.sample(self.movies, min(num_recommendations, len(self.movies)))

        # Get movie index
        idx = self.movie_indices[movie_id]

        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity[idx]))

        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:num_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]

        # Return recommended movies
        return [self.movies[i] for i in movie_indices]

    def recommend_collaborative(self, user_id, num_recommendations=6):
        """Collaborative filtering recommendation based on user similarity"""
        # If user not found, return random recommendations
        if user_id not in self.user_indices:
            return random.sample(self.movies, min(num_recommendations, len(self.movies)))

        # Get user index
        user_idx = self.user_indices[user_id]

        # Find similar users
        similar_users = list(enumerate(self.user_similarity[user_idx]))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
        similar_users = similar_users[1:4]  # Top 3 similar users

        # Find movies that similar users rated highly but the user hasn't watched
        user_ratings = self.user_item_matrix[user_idx]
        unwatched = np.where(user_ratings == 0)[0]

        # Calculate predicted ratings for unwatched movies
        predicted_ratings = []
        for movie_idx in unwatched:
            rating_sum = 0
            sim_sum = 0
            for user, sim in similar_users:
                if self.user_item_matrix[user, movie_idx] > 0:
                    rating_sum += self.user_item_matrix[user, movie_idx] * sim
                    sim_sum += sim

            if sim_sum > 0:
                predicted_ratings.append((movie_idx, rating_sum / sim_sum))

        # Sort by predicted rating
        predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
        recommended_indices = [i[0] for i in predicted_ratings[:num_recommendations]]

        # Return recommended movies
        return [self.movies[i] for i in recommended_indices if i < len(self.movies)]

    def recommend_hybrid(self, user_id=None, movie_id=None, num_recommendations=6):
        """Hybrid recommendation combining content-based and collaborative filtering"""
        if not user_id and not movie_id:
            # No user or movie specified, return trending/popular
            return random.sample(self.movies, min(num_recommendations, len(self.movies)))

        if user_id and movie_id and user_id in self.user_indices and movie_id in self.movie_indices:
            # Both user and movie specified - combine recommendations
            content_recs = self.recommend_content_based(movie_id, num_recommendations // 2)
            collab_recs = self.recommend_collaborative(user_id, num_recommendations // 2)

            # Combine and deduplicate
            hybrid_recs = content_recs + collab_recs
            seen_ids = set()
            unique_recs = []
            for rec in hybrid_recs:
                if rec['id'] not in seen_ids:
                    seen_ids.add(rec['id'])
                    unique_recs.append(rec)

            return unique_recs[:num_recommendations]

        elif user_id and user_id in self.user_indices:
            # Only user specified
            return self.recommend_collaborative(user_id, num_recommendations)

        elif movie_id and movie_id in self.movie_indices:
            # Only movie specified
            return self.recommend_content_based(movie_id, num_recommendations)

        # Fallback to random recommendations
        return random.sample(self.movies, min(num_recommendations, len(self.movies)))