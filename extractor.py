# load all songs, extract files, put in json files

import os 
import pickle

import numpy as np 
import librosa


def load_files(audio_dir):
    sr = 22050
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

    print(f"Found {len(audio_files)} songs:")
    for file in audio_files:
        print(f"  - {file}")

    # Load each file
    songs = {}
    for filename in audio_files:
        filepath = os.path.join(audio_dir, filename)
        print(f"\nLoading {filename}...")
        
        y, sr = librosa.load(filepath, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        songs[filename] = {
            'audio': y, 
            'sr': sr, 
            'duration': duration,
            'chunks': []  # Will store chunk features here
        }
        
        print(f"Loaded: {duration:.2f} seconds, {len(y)} samples")

    print(f"\nAll {len(songs)} songs loaded!")

    return songs


def extract(songs, beats_per_chunk):
    """
    Extract features using beat-based chunking instead of fixed time intervals.
    
    Args:
        songs: Dictionary of song data
        beats_per_chunk: How many beats to include in each chunk (depends on time signature)
    """
    # Process each song
    for song_name, song_data in songs.items():
        print(f"\nProcessing {song_name}...")
        
        y = song_data['audio']
        sr = song_data['sr']
        
        # STEP 1: Detect ALL beats in the song first
        print(f"  Detecting beats for entire song...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Convert beat frames to sample indices
        beat_samples = librosa.frames_to_samples(beat_frames)
        
        # Store global tempo info
        song_data['global_tempo'] = np.asarray(tempo).item()
        song_data['total_beats'] = len(beat_samples)
        
        print(f"  Found {len(beat_samples)} beats at {tempo} BPM")
        
        # STEP 2: Group beats into chunks
        num_chunks = len(beat_samples) // beats_per_chunk
        
        if num_chunks == 0:
            print(f"  Warning: Only {len(beat_samples)} beats found, need at least {beats_per_chunk}. Skipping song.")
            continue
        
        print(f"  Creating {num_chunks} chunks ({beats_per_chunk} beats each)...")
        
        # STEP 3: Extract features for each beat-aligned chunk
        for i in range(num_chunks):
            # Get the beat indices for this chunk
            beat_start_idx = i * beats_per_chunk
            beat_end_idx = beat_start_idx + beats_per_chunk
            
            # Get sample boundaries from beat positions
            start_sample = beat_samples[beat_start_idx]
            # Use the next beat as the end, or end of song for last chunk
            if beat_end_idx < len(beat_samples):
                end_sample = beat_samples[beat_end_idx]
            else:
                end_sample = len(y)
            
            # Extract the chunk
            chunk = y[start_sample:end_sample]
            
            # Skip if chunk is too short
            if len(chunk) < sr * 0.1:  # Less than 0.1 seconds
                continue
            
            # Dictionary to store this chunk's features
            chunk_features = {
                'chunk_index': i,
                'start_time': start_sample / sr,
                'end_time': end_sample / sr,
                'start_beat': beat_start_idx,
                'end_beat': beat_end_idx,
                'num_beats': beats_per_chunk
            }
            
            # 1. RMS (loudness)
            rms = librosa.feature.rms(y=chunk)
            chunk_features['rms_mean'] = np.mean(rms)
            chunk_features['rms_std'] = np.std(rms)
            
            # 2. Tempo (use global tempo since we already computed it)
            chunk_features['tempo'] = song_data['global_tempo']
            
            # 3. MFCCs (timbre)
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            chunk_features['mfcc_mean'] = np.mean(mfcc, axis=1)  # 13 values
            chunk_features['mfcc_std'] = np.std(mfcc, axis=1)    # 13 values
            
            # 4. MFCC Delta (change in timbre)
            mfcc_delta = librosa.feature.delta(mfcc)
            chunk_features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)  # 13 values
            
            # 5. Chroma (pitch/harmony)
            chroma = librosa.feature.chroma_cqt(y=chunk, sr=sr)
            chunk_features['chroma_mean'] = np.mean(chroma, axis=1)  # 12 values
            
            # 6. Spectral Centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
            chunk_features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            chunk_features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            # 7. Spectral Bandwidth (frequency spread)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
            chunk_features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            chunk_features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # 8. Band Energy Ratio (bass vs treble)
            S = np.abs(librosa.stft(chunk))
            freqs = librosa.fft_frequencies(sr=sr)
            split_freq = 2000
            split_bin = np.argmin(np.abs(freqs - split_freq))
            low_energy = np.sum(S[:split_bin, :])
            high_energy = np.sum(S[split_bin:, :])
            chunk_features['band_energy_ratio'] = low_energy / (high_energy + 1e-10)
            
            # Store this chunk's features
            song_data['chunks'].append(chunk_features)
        
        print(f"  Extracted features for {len(song_data['chunks'])} beat-aligned chunks")

    return songs