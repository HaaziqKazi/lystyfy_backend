# load all songs, extract files, put in json files

import os 
import pickle

import numpy as np 
import librosa


def load_files(audio_dir):
    sr = 22050
    # audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
    audio_files = ['adreneline.mp3', 'night_of_fire.mp3']

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


def extract(songs):
    # Process each song
    for song_name, song_data in songs.items():
        print(f"\nProcessing {song_name}...")
        
        y = song_data['audio']
        sr = song_data['sr']
        
        # Chunking parameters
        chunk_duration = 3  # seconds
        chunk_samples = chunk_duration * sr
        num_chunks = len(y) // chunk_samples
        
        print(f"  Creating {num_chunks} chunks...")
        
        # Extract features for each chunk
        for i in range(num_chunks):
            # Get chunk
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = y[start:end]
            
            # Dictionary to store this chunk's features
            chunk_features = {
                'chunk_index': i,
                'start_time': start / sr,
                'end_time': end / sr
            }
            
            # 1. RMS (loudness)
            rms = librosa.feature.rms(y=chunk)
            chunk_features['rms_mean'] = np.mean(rms)
            chunk_features['rms_std'] = np.std(rms)
            
            # 2. Tempo
            tempo, beat_frames = librosa.beat.beat_track(y=chunk, sr=sr)
            chunk_features['tempo'] = np.asarray(tempo).item()  # Extract scalar properly
            chunk_features['num_beats'] = len(beat_frames)
            
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
        
        print(f"Extracted features for {len(song_data['chunks'])} chunks")

    return songs