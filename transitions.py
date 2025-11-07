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
        tuple: (best_score, best_prev_chunk_idx, best_next_chunk_idx)
    """
    # Handle edge cases
    actual_prev_n = min(prev_n, len(prev_song_chunks))
    actual_next_n = min(next_n, len(next_song_chunks))
    
    # Get the relevant chunks
    prev_chunks = prev_song_chunks[-actual_prev_n:]  # Last N chunks
    next_chunks = next_song_chunks[:actual_next_n]    # First N chunks
    
    # Convert to vectors and normalize
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
    
    # Find best similarity and track which chunks
    max_similarity = -1
    best_prev_idx = None
    best_next_idx = None
    
    for i, vec1 in enumerate(prev_vectors):
        for j, vec2 in enumerate(next_vectors):
            sim = cosine_similarity([vec1], [vec2])[0][0]
            if sim > max_similarity:
                max_similarity = sim
                best_prev_idx = len(prev_song_chunks) - actual_prev_n + i  # Actual index in full song
                best_next_idx = j  # Index from start
    
    return max_similarity, best_prev_idx, best_next_idx


def compute_transition_matrix(songs, scaler):
    """
    Compute N×N matrix of all transition scores
    
    Returns:
        matrix: numpy array (N, N)
        song_names: list of song names in order
    """
    song_names = list(songs.keys())
    n = len(song_names)
    
    transition_matrix = np.zeros((n, n))
    # NEW: Store cut point information
    cut_points = {}  # Will store (prev_idx, next_idx) for each transition
    
    for i, prev_song in enumerate(song_names):
        for j, next_song in enumerate(song_names):
            if i == j:
                transition_matrix[i][j] = -1
            else:
                score, prev_idx, next_idx = score_transition(  # Now gets 3 values!
                    songs[prev_song]['chunks'],
                    songs[next_song]['chunks'],
                    scaler=scaler
                )
                transition_matrix[i][j] = score
                cut_points[(i, j)] = (prev_idx, next_idx)  # Store cut info
    
    return transition_matrix, song_names, cut_points

def greedy_playlist(transition_matrix, song_names, start_idx=0, length=None):
    if length is None:
        length = len(song_names)
    
    playlist = [start_idx]
    visited = {start_idx}
    current = start_idx
    
    for _ in range(length - 1):
        # Get scores from current song
        scores = transition_matrix[current].copy()
        
        # Mask visited songs
        for v in visited:
            scores[v] = -np.inf
        
        # Pick best unvisited
        next_song = np.argmax(scores)
        
        if scores[next_song] == -np.inf:
            break  # No more songs
        
        playlist.append(next_song)
        visited.add(next_song)
        current = next_song
    
    return playlist