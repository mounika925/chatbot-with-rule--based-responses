import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'John Wick', 'The Notebook', 'Avengers', 'Iron Man'],
    'description': [
        'A computer hacker learns about the true nature of reality.',
        'An ex-hit-man comes out of retirement to track down gangsters.',
        'A romantic story between two young lovers.',
        'Superheroes team up to save the world.',
        'A wealthy man becomes a superhero using technology.'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# TF-IDF Vectorization of descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def recommend(title):
    # Get index of the movie
    idx = df[df['title'] == title].index[0]
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort movies based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get top 3 similar movies (excluding itself)
    sim_scores = sim_scores[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Example usage
movie_name = 'Iron Man'
print(f"Movies similar to '{movie_name}':")
print(recommend(movie_name))
