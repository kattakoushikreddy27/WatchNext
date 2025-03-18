import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MovieRecommender:

    def __init__(self):
        self.movies_df = pd.read_csv('static/data/movies.csv')
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Combine genre and description for content-based filtering
        self.movies_df['content'] = self.movies_df['genre'] + ' ' + self.movies_df['description']
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Mood to genre/keyword mapping
        self.mood_mapping = {
            'happy': ['Comedy', 'Adventure'],
            'romantic': ['Romance', 'Drama'],
            'inspired': ['Sports', 'Drama'],
            'nostalgic': ['Drama', 'Romance'],
            'excited': ['Action', 'Adventure'],
            'thoughtful': ['Drama', 'Thriller'],
            'patriotic': ['Drama', 'Sports']
        }

    def get_movie_by_id(self, movie_id):
        return self.movies_df[self.movies_df['id'] == movie_id].to_dict('records')[0]

    def search_movies(self, query=None, genre=None):
        results = self.movies_df.copy()

        if query:
            mask = (
                results['title'].str.contains(query, case=False) |
                results['description'].str.contains(query, case=False)
            )
            results = results[mask]

        if genre:
            results = results[results['genre'] == genre]

        return results.to_dict('records')

    def get_featured_movies(self, n=6):
        return self.movies_df.nlargest(n, 'rating').to_dict('records')

    def get_all_genres(self):
        return self.movies_df['genre'].unique().tolist()

    def get_mood_recommendations(self, mood, n=6):
        """Get movie recommendations based on user's current mood."""
        if mood.lower() not in self.mood_mapping:
            return self.get_featured_movies(n)
        preferred_genres = self.mood_mapping[mood.lower()]
        mood_movies = self.movies_df[self.movies_df['genre'].isin(preferred_genres)]
        # Sort by rating and return top N movies
        return mood_movies.nlargest(n, 'rating').to_dict('records')

    def get_similar_movies(self, movie_id, n=5):
        """Find similar movies based on content similarity."""
        if movie_id not in self.movies_df['id'].values:
            raise IndexError("Movie ID not found")

        # Get the index of the movie in the DataFrame
        idx = self.movies_df.index[self.movies_df['id'] == movie_id][0]

        # Get pairwise similarity scores for that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort movies based on similarity scores (descending order)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top N similar movies (skip the first because it's the same movie)
        sim_scores = sim_scores[1:n + 1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top similar movies
        return self.movies_df.iloc[movie_indices].to_dict('records')

    def get_available_moods(self):
        """Get list of available moods for recommendations."""
        return list(self.mood_mapping.keys())


# Example Usage:
# recommender = MovieRecommender()
# print(recommender.search_movies(query="Avengers"))
# print(recommender.get_mood_recommendations("happy"))
# print(recommender.get_featured_movies())
