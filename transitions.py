from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from vectorise import features_to_vector

def score_transition(prev_song_chunks, next_song_chunks, scaler, prev_n=3, next_n=3):
    """
    Score how well song A transitions into song B
    
    Args:
        prev_song_chunks: List of chunk feature dicts from song A
        next_song_chunks: List of chunk feature dicts from song B
        scaler: Fitted StandardScaler
        prev_n: How many ending chunks of A to consider
        next_n: How many starting chunks of B to consider
    
    Returns:
        float: Best similarity score (0 to 1)
    """
    # Handle edge cases
    actual_prev_n = min(prev_n, len(prev_song_chunks))
    actual_next_n = min(next_n, len(next_song_chunks))
    
    # Get the relevant chunks
    prev_chunks = prev_song_chunks[-actual_prev_n:]  # Last N chunks
    next_chunks = next_song_chunks[:actual_next_n]    # First N chunks
    
    # Convert to vectors and normalize (do this once, not in loop!)
    prev_vectors = []
    for chunk in prev_chunks:
        vec = features_to_vector(chunk)
        vec_norm = scaler.transform([vec])[0]
        prev_vectors.append(vec_norm)
    
    next_vectors = []
    for chunk in next_chunks:
        vec = features_to_vector(chunk)
        vec_norm = scaler.transform([vec])[0]
        next_vectors.append(vec_norm)
    
    # Find best similarity
    max_similarity = -1
    for vec1 in prev_vectors:
        for vec2 in next_vectors:
            sim = cosine_similarity([vec1], [vec2])[0][0]
            max_similarity = max(max_similarity, sim)


def compute_transition_matrix(songs, scaler):
    """
    Compute N×N matrix of all transition scores
    
    Returns:
        matrix: numpy array (N, N)
        song_names: list of song names in order
    """
    song_names = list(songs.keys())
    n = len(song_names)
    
    # Initialize matrix with zeros
    transition_matrix = np.zeros((n, n))
    
    print(f"Computing {n} x {n} transition matrix...")
    
    for i, prev_song in enumerate(song_names):
        for j, next_song in enumerate(song_names):
            if i == j:
                # Don't transition to same song
                transition_matrix[i][j] = -1
            else:
                score = score_transition(
                    songs[prev_song]['chunks'],
                    songs[next_song]['chunks'],
                    scaler=scaler
                )
                transition_matrix[i][j] = score
                print(f"{prev_song} -> {next_song}: {score:.3f}")
    
    print("Matrix complete!")
    return transition_matrix, song_names

    # and then find the best order - this is a directed cyclic graph,
    # and find the best path with biggest distance (travelling salesperson problem)
