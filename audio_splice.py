from pydub import AudioSegment
import numpy as np

class AudioSplicer:
    def __init__(self, crossfade_duration=2000):  # milliseconds
        """
        crossfade_duration: How long the crossfade should be (in ms)
        """
        self.crossfade_duration = crossfade_duration
    
    def load_audio(self, filepath):
        """Load audio file"""
        return AudioSegment.from_file(filepath)
    
    def splice_two_songs(self, song_a_path, song_b_path, cut_a_time, start_b_time):
        """
        Splice two songs together with crossfade
        
        Args:
            song_a_path: Path to first song
            song_b_path: Path to second song
            cut_a_time: Where to cut song A (seconds)
            start_b_time: Where to start song B (seconds)
        
        Returns:
            AudioSegment: Combined audio
        """
        # Load songs
        audio_a = self.load_audio(song_a_path)
        audio_b = self.load_audio(song_b_path)
        
        # Convert seconds to milliseconds (pydub uses ms)
        cut_a_ms = int(cut_a_time * 1000)
        start_b_ms = int(start_b_time * 1000)
        
        # Cut the songs
        song_a_cut = audio_a[:cut_a_ms]
        song_b_cut = audio_b[start_b_ms:]
        
        # Crossfade them together
        spliced = song_a_cut.append(song_b_cut, crossfade=self.crossfade_duration)
        
        return spliced
    
    def render_playlist(songs_dict, playlist, song_names, cut_points, output_path="playlist.mp3"):
        """
        Render playlist by building up the mix song by song
        """
        
        print("Rendering playlist...")
        
        combined = None
        
        for i in range(len(playlist)):
            song_idx = playlist[i]
            song_name = song_names[song_idx]
            song_path = f"data/Audio/{song_name}"
            
            # Load song
            audio = AudioSegment.from_file(song_path)
            
            # Determine cut points for this song
            if i == 0:
                # First song: keep from start, cut at end
                if i < len(playlist) - 1:  # If there's a next song
                    next_song_idx = playlist[i + 1]
                    end_chunk_idx, _ = cut_points[(song_idx, next_song_idx)]
                    cut_time = songs_dict[song_name]['chunks'][end_chunk_idx]['end_time']
                    cut_ms = int(cut_time * 1000)
                    audio = audio[:cut_ms]  # Keep beginning, cut end
                # else: last song, keep full
                
                combined = audio
                
            elif i == len(playlist) - 1:
                # Last song: cut beginning, keep to end
                prev_song_idx = playlist[i - 1]
                _, start_chunk_idx = cut_points[(prev_song_idx, song_idx)]
                start_time = songs_dict[song_name]['chunks'][start_chunk_idx]['start_time']
                start_ms = int(start_time * 1000)
                audio = audio[start_ms:]  # Cut beginning, keep end
                
                combined = combined.append(audio, crossfade=2000)
                
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
                
                audio = audio[start_ms:end_ms]  # Cut both ends!
                
                combined = combined.append(audio, crossfade=2000)
            
            print(f"{i+1}. {song_name}")
        
        # Export
        combined.export(output_path, format="mp3", bitrate="320k")
        print(f"Saved to {output_path}")