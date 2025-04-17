import os 
import numpy as np 
import pandas as pd
import matplotlib as plt
import librosa
import scipy as sp # install

# copy relative path
debussy_file = 'debussy.wav'
debussy, sr = librosa.load(debussy_file)

'''
Plan - need to extract key, pitch, timbre, chords, idk what else 

beats and and important rhythms - use amplitude envelope to extract (helps with tempo analysis)
loudness/softness - need to match 'intensity' in transitions between songs (use loudness?)
single loudness measure per segment (RMS energy) or a continuous amplitude trace over time (amp. envelope)
pitch / key - audacity estimates it by the first note, otherwise it's hard if its not known idk throw FFT lmao
mel spectrogram / Mel frequency cepstral coeff as an input feature in Music Classification, for identifying instruments
spectral centroid / bandwidth / band energy ratio


lwk can first use shit where notes and stuff is already known, and then train 
how to measure fitness? maybe don't even need a learning network just music theory thing


then a vector database thing for text on the song (each song has words associated - then match them)

'''