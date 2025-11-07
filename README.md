make something like a language model for music. 
output the probability of two audio clips being in sequence! 
we train on cut up song clips labeled 1, as in they are definitely in sequence, and mismatched song clips labeled 0


Plan - need to extract key, pitch, timbre, chords, idk what else 

beats and important rhythms - use amplitude envelope to extract (helps with tempo analysis)
loudness/softness - need to match 'intensity' in transitions between songs (use loudness?)
single loudness measure per segment (RMS energy) or a continuous amplitude trace over time (amp. envelope)
pitch / key - audacity estimates it by the first note, otherwise it's hard if its not known idk throw FFT lmao
mel spectrogram / Mel frequency cepstral coeff as an input feature in Music Classification, for identifying instruments
spectral centroid / bandwidth / band energy ratio

autoencoders - generates music lol
variational AE - generates music... but fancier 
LSTM too 
how is Background Noise Removal done?

lwk don't get too hung up on ts just start making the program lol 


lwk can first use shit where notes and stuff is already known, and then train 
how to measure fitness? maybe don't even need a learning network just music theory thing


then a vector database thing for text on the song (each song has words associated - then match them)

maybe like a personalized radio jockey
uk this has applications in personalized advertisement targeting. LLM + ad + smooth transition. the best ads are the ones you don't even notice are there 





optional for band energy ratio: 

```python
# Add this helper function at the top of your file
def calculate_split_frequency_bin(S, split_frequency, sample_rate):
    """Calculate which frequency bin corresponds to split_frequency"""
    frequency_range = sample_rate / 2  # Nyquist frequency
    frequency_delta_per_bin = frequency_range / S.shape[0]
    split_frequency_bin = int(np.floor(split_frequency / frequency_delta_per_bin))
    return split_frequency_bin

# Then in your feature extraction:
split_bin = calculate_split_frequency_bin(S, 2000, sr)
low_energy = np.sum(S[:split_bin, :])
high_energy = np.sum(S[split_bin:, :])
```

ways to improve: 
do it on beats -> because you want to transition on each beat, not on a random second (because it may cut)
put the features into a json file (so to not recompute every single time)
a lot of the songs end or start with silence lmao
genetic algorithm



EEG - use brain waves to recommend?

also go visualise lmao