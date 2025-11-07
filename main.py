import os 
import pickle

import numpy as np 
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from vectorise import features_to_vector    
from transitions import score_transition, compute_transition_matrix, greedy_playlist

from extractor import load_files, extract


songs = load_files(audio_dir = 'data/Audio')

songs = extract(songs)


# Collect all chunk vectors first
all_vectors = []
for song_name, song_data in songs.items():
    for chunk in song_data['chunks']:
        all_vectors.append(features_to_vector(chunk))

all_vectors = np.array(all_vectors)

# Fit scaler on ALL chunks
scaler = StandardScaler()
scaler.fit(all_vectors)

matrix, song_names, cut_points = compute_transition_matrix(songs, scaler=scaler)

# Save it
np.save('data/transition_matrix.npy', matrix)
with open('data/song_names.pkl', 'wb') as f:
    pickle.dump(song_names, f)

print(f"\n{matrix}")
print(f"\nSong order: {song_names}")

playlist = greedy_playlist(matrix, song_names)


for i in range(len(playlist) - 1):
    song_a_idx = playlist[i]
    song_b_idx = playlist[i + 1]
    
    # Get the cut points
    prev_chunk_idx, next_chunk_idx = cut_points[(song_a_idx, song_b_idx)]
    
    # Get timestamps
    song_a_name = song_names[song_a_idx]
    song_b_name = song_names[song_b_idx]
    
    cut_a = songs[song_a_name]['chunks'][prev_chunk_idx]['end_time']
    start_b = songs[song_b_name]['chunks'][next_chunk_idx]['start_time']
    
    print(f"{song_a_name} (cut at {cut_a:.1f}s) → {song_b_name} (start at {start_b:.1f}s)")