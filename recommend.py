import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

from extract_features import load_saved_features
from vectorise import features_to_vector
from transitions import compute_transition_matrix, greedy_playlist


def recommend_playlist(features_dir='data/features', start_song=None, playlist_length=None):
    """
    Generate a recommended playlist from pre-extracted features
    
    Args:
        features_dir: Directory containing saved features
        start_song: Optional song name to start playlist with
        playlist_length: Optional length of playlist (default: all songs)
        
    Returns:
        playlist: List of song indices
        song_names: List of song names
        songs: Dictionary of song features
        cut_points: Dictionary of cut point information
    """
    
    # Load pre-extracted features
    print(f"\nLoading features from: {features_dir}/")
    songs, song_index = load_saved_features(features_dir)
    
    print("\n" + "=" * 60)
    print("Computing transition scores...")
    print("=" * 60)
    
    # Collect all chunk vectors
    print("\nVectorizing chunks...")
    all_vectors = []
    for song_name, song_data in songs.items():
        for chunk in song_data['chunks']:
            all_vectors.append(features_to_vector(chunk))
    
    all_vectors = np.array(all_vectors)
    print(f"Total vectors: {len(all_vectors)}")
    
    # Fit scaler on ALL chunks
    print("\nNormalizing features")
    scaler = StandardScaler()
    scaler.fit(all_vectors)
    
    # Compute transition matrix
    print("\nComputing transition matrix")
    matrix, song_names, cut_points = compute_transition_matrix(songs, scaler=scaler)
    
    print(f"\nTransition matrix shape: {matrix.shape}")
    print(f"Songs: {len(song_names)}")
    
    # Determine start index
    start_idx = 0
    if start_song:
        if start_song in song_names:
            start_idx = song_names.index(start_song)
            print(f"\nStarting with: {start_song}")
        else:
            print(f"\nSong '{start_song}' not found. Starting with: {song_names[0]}")
    
    # Generate playlist
    print("\n" + "=" * 60)
    print("Generating optimal playlist...")
    print("=" * 60)
    
    playlist = greedy_playlist(
        matrix, 
        song_names, 
        start_idx=start_idx, 
        length=playlist_length
    )
    
    return playlist, song_names, songs, cut_points, matrix


def print_playlist(playlist, song_names, songs, cut_points):
    """
    Print the generated playlist with transition details
    """
    print("\n" + "=" * 60)
    print("RECOMMENDED PLAYLIST")
    print("=" * 60)
    
    total_duration = 0
    
    for i, song_idx in enumerate(playlist):
        song_name = song_names[song_idx]
        song_duration = songs[song_name]['duration']
        
        print(f"\n{i+1}. {song_name}")
        print(f"   Duration: {song_duration:.1f}s")
        print(f"   Tempo: {songs[song_name].get('global_tempo', 'N/A')} BPM")
        
        if i < len(playlist) - 1:
            # Show transition info
            next_idx = playlist[i + 1]
            next_name = song_names[next_idx]
            
            prev_chunk_idx, next_chunk_idx = cut_points[(song_idx, next_idx)]
            
            cut_time = songs[song_name]['chunks'][prev_chunk_idx]['end_time']
            start_time = songs[next_name]['chunks'][next_chunk_idx]['start_time']
            
            print(f"   Transition at {cut_time:.1f}s")
            print(f"   (beat {songs[song_name]['chunks'][prev_chunk_idx]['end_beat']})")
            
            # Calculate actual duration played for this song
            played_duration = cut_time
        else:
            # Last song plays to the end
            played_duration = song_duration
        
        total_duration += played_duration
    
    print("\n" + "=" * 60)
    print(f"Total playlist duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print("=" * 60)


def save_playlist(playlist, song_names, songs, cut_points, output_file='playlist_output.txt'):
    """
    Save the playlist to a text file
    """
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RECOMMENDED PLAYLIST\n")
        f.write("=" * 60 + "\n\n")
        
        for i, song_idx in enumerate(playlist):
            song_name = song_names[song_idx]
            song_duration = songs[song_name]['duration']
            
            f.write(f"{i+1}. {song_name}\n")
            f.write(f"   Duration: {song_duration}s\n")
            f.write(f"   Tempo: {songs[song_name].get('global_tempo', 'N/A')} BPM\n")
            
            if i < len(playlist) - 1:
                next_idx = playlist[i + 1]
                next_name = song_names[next_idx]
                prev_chunk_idx, next_chunk_idx = cut_points[(song_idx, next_idx)]
                cut_time = songs[song_name]['chunks'][prev_chunk_idx]['end_time']
                
                f.write(f"   Cut at {cut_time}s → {next_name}\n")
            
            f.write("\n")
    
    print(f"\nPlaylist saved to: {output_file}")


def save_transition_data(matrix, song_names, cut_points, output_dir='data/playlists'):
    """
    Save transition matrix and cut points for future use
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save transition matrix
    matrix_file = os.path.join(output_dir, 'transition_matrix.npy')
    np.save(matrix_file, matrix)
    
    # Save song names
    names_file = os.path.join(output_dir, 'song_names.pkl')
    with open(names_file, 'wb') as f:
        pickle.dump(song_names, f)
    
    # Save cut points
    cuts_file = os.path.join(output_dir, 'cut_points.pkl')
    with open(cuts_file, 'wb') as f:
        pickle.dump(cut_points, f)
    
    print(f"\nTransition data saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate playlist recommendations from pre-extracted features'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='data/features',
        help='Directory containing extracted features (default: data/features)'
    )
    parser.add_argument(
        '--start_song',
        type=str,
        help='Song to start playlist with (optional)'
    )
    parser.add_argument(
        '--length',
        type=int,
        help='Length of playlist (default: all songs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/playlists/playlist.txt',
        help='Output file for playlist (default: data/playlists/playlist.txt)'
    )
    parser.add_argument(
        '--save_transitions',
        action='store_true',
        help='Save transition matrix and cut points for later use'
    )
    
    args = parser.parse_args()
    
    # Check if features exist
    index_file = os.path.join(args.features_dir, 'song_index.json')
    if not os.path.exists(index_file):
        print(f"\nNo features found at {args.features_dir}/")
        print("Please run 'python extract_features.py' first!")
        return
    
    # Generate playlist
    playlist, song_names, songs, cut_points, matrix = recommend_playlist(
        features_dir=args.features_dir,
        start_song=args.start_song,
        playlist_length=args.length
    )
    
    # Print results
    print_playlist(playlist, song_names, songs, cut_points)
    
    # Save playlist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_playlist(playlist, song_names, songs, cut_points, output_file=args.output)
    
    # Optionally save transition data
    if args.save_transitions:
        save_transition_data(matrix, song_names, cut_points)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()