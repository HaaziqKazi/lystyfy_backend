import os 
import numpy as np 
import pandas as pd
import matplotlib as plt
import librosa

# copy relative path
debussy_file = 'debussy.wav'
debussy, sr = librosa.load(debussy_file)

