# Import necessary libraries
import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy
import os
import pandas as pd
# from torchvision.io import read_image
from torch.utils.data import Dataset

#### Define worker classes ########
#   ...
#### Worker Functions #########

# Test and notify that this module could be imported successfully
def test_import():
    print("Successfully imported this module")

# Function to read a WAV file
def read_wav(file_path, mode='rb'):
    # Open the WAV file in the specified mode (default is read binary)
    wav_obj = wave.open(file_path, mode)
    # Return the WAV object
    return wav_obj

# Decode the audio samples from a wav object and check if wave is mono or not.
def decode_wav(wav_obj):
    # Extract Raw Audio from Wav File
    signal = wav_obj.readframes(-1)
    signal = np.fromstring(signal, np.int16)

    # If Stereo
    if wav_obj.getnchannels() < 2:
        print("I: Just mono files")
    return signal
    

# Function to save data as a WAV file
def save_wav(file_path, data=None, sample_rate=44100):
    # Check if the data is not None before saving

    # Open the WAV file in write mode
    with wave.open(file_path, 'w') as wav_file:
        # Convert the data to 16-bit integer values. This ensures that the 
        # highest value is within the 16-bit range which is the usual depth for WAV files.
        audio = np.int16(data * 32767)
        
        # Set the parameters for the WAV file: 
        # (number of channels, width in bytes of samples, sample rate, 
        # number of frames, compression type, compression name)
        wav_file.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
        print('D:',  audio)
        
        # Write the audio data to the WAV file
        wav_file.writeframes(audio.tobytes())

# TODO: Get fft of wav data
def wav_fft():
    return None




