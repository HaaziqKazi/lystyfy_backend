import os 
import numpy as np 
import pandas as pd
#import pytorch as pt

import matplotlib as plt
import librosa
import scipy as sp # install



'''class MusicClassifier(pt.nn.module):
    def __init__(self):
        super().__init__()'''

investigations_file = 'Audio/investigations.mp3'
sneaky_adventure_file = 'Audio/sneaky-adventure.mp3'

investigations, sr1 = librosa.load(investigations_file) # load the audio file, returns audio time series and sampling rate
sneaky_adventure, sr2 = librosa.load(sneaky_adventure_file)
investigations_duration = librosa.get_duration(y=investigations, sr=22050) # get the duration of the audio file
sneaky_adventure_duration = librosa.get_duration(y=sneaky_adventure, sr=22050)
print(f'Investigations duration: {investigations_duration} seconds')
print(f'Sneaky Adventure duration: {sneaky_adventure_duration} seconds')

investigations_tempo, investigations_beats = librosa.beat.beat_track(y=investigations, sr=sr1)
print(investigations_tempo, investigations_beats)

librosa.feature.chroma_stft

'''
make something like a language model for music. 
output the probability of two audio clips being in sequence! 
we train on cut up song clips labeled 1, as in they are definitely in sequence, and mismatched song clips labeled 0


Plan - need to extract key, pitch, timbre, chords, idk what else 

beats and and important rhythms - use amplitude envelope to extract (helps with tempo analysis)
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

hi!!! hello
'''