import os
from flask import Flask, render_template, request, flash
from recommender import MovieRecommender

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Initialize recommender
recommender = MovieRecommender()


@app.route('/')
def index():
    featured_movies = recommender.get_featured_movies()
    available_moods = recommender.get_available_moods()
    return render_template('index.html',
                           featured_movies=featured_movies,
                           available_moods=available_moods)


@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    try:
        movie = recommender.get_movie_by_id(movie_id)
        similar_movies = recommender.get_similar_movies(movie_id)
        return render_template('movie.html', movie=movie, similar_movies=similar_movies)
    except IndexError:
        flash('Movie not found')
        return render_template('index.html'), 404


@app.route('/search')
def search():
    query = request.args.get('q', '')
    genre = request.args.get('genre', '')

    movies = recommender.search_movies(query, genre)
    genres = recommender.get_all_genres()

    return render_template('search.html',
                           movies=movies,
                           query=query,
                           genres=genres,
                           selected_genre=genre)

@app.route('/mood-matcher')
def mood_matcher():
    mood = request.args.get('mood', '')
    available_moods = recommender.get_available_moods()
    recommendations = []
    if mood:
        recommendations = recommender.get_mood_recommendations(mood)
    return render_template('mood.html',
                         available_moods=available_moods,
                         selected_mood=mood,
                         recommendations=recommendations)


@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html'), 500
