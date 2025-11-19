import os
import json
import pickle
import argparse
from datetime import datetime

import numpy as np
from extractor import load_files, extract


def save_features(songs, output_dir='data/features'):
    """
    Save extracted song features to disk
    
    Args:
        songs: Dictionary of song data with extracted features
        output_dir: Directory to save features to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    song_index = {
        'extraction_date': datetime.now().isoformat(),
        'songs': {},
        'metadata': {}
    }
    
    print(f"\nSaving features to {output_dir}/")
    
    for song_name, song_data in songs.items():
        print(f"Saving {song_name}...")
        
        # Create a clean filename (remove .mp3 extension, replace special chars)
        clean_name = song_name.replace('.mp3', '').replace(' ', '_')
        feature_file = os.path.join(output_dir, f"{clean_name}_features.pkl")
        
        # Prepare song data for saving (exclude raw audio to save space)
        song_features = {
            'song_name': song_name,
            'sr': song_data['sr'],
            'duration': song_data['duration'],
            'global_tempo': song_data.get('global_tempo'),
            'total_beats': song_data.get('total_beats'),
            'chunks': song_data['chunks']
        }
        
        # Save to pickle file
        with open(feature_file, 'wb') as f:
            pickle.dump(song_features, f)
        
        # Add to index
        song_index['songs'][song_name] = {
            'feature_file': feature_file,
            'clean_name': clean_name,
            'duration': song_data['duration'],
            'tempo': song_data.get('global_tempo'),
            'num_chunks': len(song_data['chunks'])
        }
        
        # Store metadata
        song_index['metadata'][song_name] = {
            'duration': song_data['duration'],
            'sample_rate': song_data['sr'],
            'tempo_bpm': song_data.get('global_tempo'),
            'total_beats': song_data.get('total_beats'),
            'num_chunks': len(song_data['chunks'])
        }
    
    # Save the index file
    index_file = os.path.join(output_dir, 'song_index.json')
    with open(index_file, 'w') as f:
        json.dump(song_index, f, indent=2)
    
    print(f"\nSaved features for {len(songs)} songs")
    print(f"Index saved to: {index_file}")
    
    return song_index


def load_saved_features(features_dir='data/features'):
    """
    Load previously extracted features from disk
    
    Args:
        features_dir: Directory containing saved features
        
    Returns:
        Dictionary of songs with their features
    """
    # Load the index
    index_file = os.path.join(features_dir, 'song_index.json')
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No feature index found at {index_file}. Run feature extraction first!")
    
    with open(index_file, 'r') as f:
        song_index = json.load(f)
    
    print(f"Loading features from {features_dir}/")
    print(f"Features extracted on: {song_index['extraction_date']}")
    
    # Load each song's features
    songs = {}
    for song_name, song_info in song_index['songs'].items():
        feature_file = song_info['feature_file']
        
        if not os.path.exists(feature_file):
            print(f"  Warning: Feature file not found for {song_name}")
            continue
        
        with open(feature_file, 'rb') as f:
            song_features = pickle.load(f)
        
        songs[song_name] = song_features
        print(f"Loaded {song_name}: {len(song_features['chunks'])} chunks")
    
    print(f"\nLoaded {len(songs)} songs")
    
    return songs, song_index


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio features from songs and save to disk'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='data/Audio',
        help='Directory containing audio files (default: data/Audio)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/features',
        help='Directory to save extracted features (default: data/features)'
    )
    parser.add_argument(
        '--beats_per_chunk',
        type=int,
        default=8,
        help='Number of beats per chunk (default: 8)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-extraction even if features already exist'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    
    # Load audio files
    print(f"\nLoading audio files from: {args.audio_dir}")
    songs = load_files(audio_dir=args.audio_dir)
    
    # Check if features already exist for all loaded songs
    index_file = os.path.join(args.output_dir, 'song_index.json')
    
    if os.path.exists(index_file) and not args.force:
        try:
            # Load existing features
            existing_songs, existing_index = load_saved_features(args.output_dir)
            existing_song_names = set(existing_songs.keys())
            current_song_names = set(songs.keys())
            
            # Check if all current songs are already extracted
            missing_songs = current_song_names - existing_song_names
            extra_songs = existing_song_names - current_song_names
            
            if not missing_songs:
                # All songs already extracted!
                print(f"\nAll songs already have extracted features in {args.output_dir}/")
                print("Use --force to re-extract, or run recommend.py to use existing features")
                
                print("\nExtracted songs:")
                for name in current_song_names:
                    meta = existing_index['metadata'][name]
                    print(f"  • {name}: {meta['duration']:.1f}s, "
                          f"{meta['tempo_bpm']:.1f} BPM, "
                          f"{meta['num_chunks']} chunks")
                
                if extra_songs:
                    print(f"\nNote: {len(extra_songs)} song(s) in features but not in audio directory:")
                    for name in extra_songs:
                        print(f"  • {name}")
                
                return
            else:
                # Some songs are missing
                print(f"\nFound existing features, but {len(missing_songs)} song(s) need extraction:")
                for name in missing_songs:
                    print(f"  • {name}")
                print("\nProceeding with extraction of missing songs...")
                
        except Exception as e:
            print(f"\nError checking existing features: {e}")
            print("Proceeding with full extraction...")
    
    # Extract features
    print(f"\nExtracting features (beats_per_chunk={args.beats_per_chunk})...")
    songs = extract(songs, beats_per_chunk=args.beats_per_chunk)
    
    # Save features
    print("\n" + "=" * 60)
    song_index = save_features(songs, output_dir=args.output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("\nSummary:")
    for name, meta in song_index['metadata'].items():
        print(f"  • {name}")
        print(f"    Duration: {meta['duration']:.1f}s")
        print(f"    Tempo: {meta['tempo_bpm']:.1f} BPM")
        print(f"    Chunks: {meta['num_chunks']}")
    
    print(f"\nFeatures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()