import os
import argparse
import pickle
import numpy as np
from pydub import AudioSegment

from extract_features import load_saved_features


def load_playlist_data(playlist_dir='data/playlists'):
    """
    Load the transition data (playlist order and cut points)
    
    Returns:
        playlist: List of song indices
        song_names: List of song names
        cut_points: Dictionary of cut point information
    """
    # Load song names
    names_file = os.path.join(playlist_dir, 'song_names.pkl')
    if not os.path.exists(names_file):
        raise FileNotFoundError(f"Song names file not found: {names_file}")
    
    with open(names_file, 'rb') as f:
        song_names = pickle.load(f)
    
    # Load cut points
    cuts_file = os.path.join(playlist_dir, 'cut_points.pkl')
    if not os.path.exists(cuts_file):
        raise FileNotFoundError(f"Cut points file not found: {cuts_file}")
    
    with open(cuts_file, 'rb') as f:
        cut_points = pickle.load(f)
    
    return song_names, cut_points


def render_playlist(songs_dict, playlist, song_names, cut_points, 
                   audio_dir='data/Audio',
                   output_path='data/output/mixed_playlist.mp3',
                   crossfade_duration=2000,
                   bitrate='320k'):
    """
    Render the playlist into a single mixed audio file
    
    Args:
        songs_dict: Dictionary of song features (with metadata)
        playlist: List of song indices in order
        song_names: List of song names
        cut_points: Dictionary mapping (song_a_idx, song_b_idx) to (chunk_a_idx, chunk_b_idx)
        audio_dir: Directory containing original audio files
        output_path: Where to save the mixed output
        crossfade_duration: Length of crossfade in milliseconds (default: 2000ms = 2 seconds)
        bitrate: Output bitrate (default: 320k)
    """
    print("=" * 60)
    print("PLAYLIST RENDERING")
    print("=" * 60)
    
    print(f"\nPlaylist order ({len(playlist)} songs):")
    for i, song_idx in enumerate(playlist):
        print(f"  {i+1}. {song_names[song_idx]}")
    
    print(f"\nCrossfade duration: {crossfade_duration}ms")
    print(f"Output bitrate: {bitrate}")
    
    # Build the mix song by song
    combined = None
    total_duration_ms = 0
    
    print("\n" + "=" * 60)
    print("Rendering")
    print("=" * 60)
    
    for i in range(len(playlist)):
        song_idx = playlist[i]
        song_name = song_names[song_idx]
        song_path = os.path.join(audio_dir, song_name)
        
        print(f"\n[{i+1}/{len(playlist)}] Processing: {song_name}")
        
        # Check if file exists
        if not os.path.exists(song_path):
            print(f"  Audio file not found: {song_path}")
            print(f"  Skipping")
            continue
        
        # Load the audio file
        print(f"  Loading audio")
        audio = AudioSegment.from_file(song_path)
        original_duration = len(audio)
        
        # Determine what portion to use
        if i == 0:
            # First song: keep from start, cut at end (if there's a next song)
            if i < len(playlist) - 1:
                next_song_idx = playlist[i + 1]
                end_chunk_idx, _ = cut_points[(song_idx, next_song_idx)]
                cut_time = songs_dict[song_name]['chunks'][end_chunk_idx]['end_time']
                cut_ms = int(cut_time * 1000)
                audio = audio[:cut_ms]
                print(f"  Using: 0.0s to {cut_time}s (cut end)")
            else:
                print(f"  Using: entire song (only song in playlist)")
            
            combined = audio
            total_duration_ms += len(audio)
            
        elif i == len(playlist) - 1:
            # Last song: cut beginning, keep to end
            prev_song_idx = playlist[i - 1]
            _, start_chunk_idx = cut_points[(prev_song_idx, song_idx)]
            start_time = songs_dict[song_name]['chunks'][start_chunk_idx]['start_time']
            start_ms = int(start_time * 1000)
            audio = audio[start_ms:]
            print(f"  Using: {start_time}s to end (cut beginning)")
            
            # Crossfade with previous
            print(f"  Crossfading with previous song ({crossfade_duration}ms)")
            combined = combined.append(audio, crossfade=crossfade_duration)
            total_duration_ms += len(audio) - crossfade_duration
            
        else:
            # Middle song: cut both beginning and end
            prev_song_idx = playlist[i - 1]
            next_song_idx = playlist[i + 1]
            
            # Where to start this song
            _, start_chunk_idx = cut_points[(prev_song_idx, song_idx)]
            start_time = songs_dict[song_name]['chunks'][start_chunk_idx]['start_time']
            start_ms = int(start_time * 1000)
            
            # Where to end this song
            end_chunk_idx, _ = cut_points[(song_idx, next_song_idx)]
            end_time = songs_dict[song_name]['chunks'][end_chunk_idx]['end_time']
            end_ms = int(end_time * 1000)
            
            audio = audio[start_ms:end_ms]
            print(f"  Using: {start_time:.1f}s to {end_time:.1f}s (cut both ends)")
            
            # Crossfade with previous
            print(f"  Crossfading with previous song ({crossfade_duration}ms)")
            combined = combined.append(audio, crossfade=crossfade_duration)
            total_duration_ms += len(audio) - crossfade_duration
        
        print(f"  Added to mix (cumulative: {total_duration_ms/1000}s)")
    
    if combined is None:
        print("\nNo audio was combined. Check that audio files exist in the audio directory.")
        return None
    
    # Export the final mix
    print("\n" + "=" * 60)
    print("Exporting final mix")
    print("=" * 60)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nExporting to: {output_path}")
    print(f"Total duration: {total_duration_ms/1000:.1f}s ({total_duration_ms/1000/60:.1f} minutes)")
    print(f"Bitrate: {bitrate}")
    
    combined.export(output_path, format="mp3", bitrate=bitrate)
    
    print(f"\nSuccessfully rendered playlist!")
    print(f"Saved to: {output_path}")
    
    return output_path


def load_playlist_from_recommend(features_dir='data/features', playlist_dir='data/playlists'):
    """
    Load a playlist generated by recommend.py
    
    Returns:
        playlist: List of song indices
        song_names: List of song names
        songs_dict: Dictionary of song features
        cut_points: Dictionary of cut points
    """
    # Load features
    print(f"Loading features from: {features_dir}/")
    songs_dict, _ = load_saved_features(features_dir)
    
    # Load playlist data
    print(f"Loading playlist from: {playlist_dir}/")
    song_names, cut_points = load_playlist_data(playlist_dir)
    
    # Load transition matrix to get playlist order
    matrix_file = os.path.join(playlist_dir, 'transition_matrix.npy')
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(
            f"Transition matrix not found: {matrix_file}\n"
            "Did you run 'python recommend.py --save_transitions'?"
        )
    
    # Read the playlist.txt file to get the order
    playlist_file = os.path.join(playlist_dir, 'playlist.txt')
    if not os.path.exists(playlist_file):
        raise FileNotFoundError(f"Playlist file not found: {playlist_file}")
    
    # Parse playlist.txt to get song order
    playlist = []
    with open(playlist_file, 'r') as f:
        for line in f:
            # Look for lines like "1. song_name.mp3"
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                # Extract song name
                song_name = line.split('. ', 1)[1].split('\n')[0].strip()
                if song_name in song_names:
                    playlist.append(song_names.index(song_name))
    
    if not playlist:
        raise ValueError("Could not parse playlist order from playlist.txt")
    
    print(f"✓ Loaded playlist with {len(playlist)} songs")
    
    return playlist, song_names, songs_dict, cut_points


def main():
    parser = argparse.ArgumentParser(
        description='Render a recommended playlist into a single mixed audio file'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='data/features',
        help='Directory containing extracted features (default: data/features)'
    )
    parser.add_argument(
        '--playlist_dir',
        type=str,
        default='data/playlists',
        help='Directory containing playlist data (default: data/playlists)'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='data/Audio',
        help='Directory containing original audio files (default: data/Audio)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/output/mixed_playlist.mp3',
        help='Output file path (default: data/output/mixed_playlist.mp3)'
    )
    parser.add_argument(
        '--crossfade',
        type=int,
        default=2000,
        help='Crossfade duration in milliseconds (default: 2000)'
    )
    parser.add_argument(
        '--bitrate',
        type=str,
        default='320k',
        help='Output bitrate (default: 320k)'
    )
    
    args = parser.parse_args()
    
    # Check if playlist data exists
    names_file = os.path.join(args.playlist_dir, 'song_names.pkl')
    if not os.path.exists(names_file):
        print(f"\nPlaylist data not found in {args.playlist_dir}/")
        print("Please run 'python recommend.py --save_transitions' first!")
        return
    
    # Load playlist
    try:
        playlist, song_names, songs_dict, cut_points = load_playlist_from_recommend(
            features_dir=args.features_dir,
            playlist_dir=args.playlist_dir
        )
    except Exception as e:
        print(f"\nError loading playlist: {e}")
        return
    
    # Render the playlist
    try:
        output_path = render_playlist(
            songs_dict=songs_dict,
            playlist=playlist,
            song_names=song_names,
            cut_points=cut_points,
            audio_dir=args.audio_dir,
            output_path=args.output,
            crossfade_duration=args.crossfade,
            bitrate=args.bitrate
        )
        
        if output_path:
            print("\n" + "=" * 60)
            print("RENDERING COMPLETE!")
            print("=" * 60)
            print(f"\n🎵 Your mixed playlist is ready!")
            print(f"📁 Location: {output_path}")
            print(f"\nYou can now play it in your favorite music player!")
            
    except Exception as e:
        print(f"\nError rendering playlist: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()