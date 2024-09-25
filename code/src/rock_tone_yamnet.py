import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd

import sys
sys.path.append('code')
from utils import tone_set_loader as tsl

### CLASSES #####

# Rock tone class definition
class YAMToneNEt():
    def __init__(self, *args, **kwargs) -> None:
        self.encoder = self.define_encoder(*args, **kwargs) 
        hub.KerasLayer('https://tfhub.dev/google/yamnet/1', trainable=False)
        self.decoder = self.define_decoder(*args, **kwargs) 
        self.optimizer = None

    def forward(self, x):
        result = self.encode(x)
        return self.decode(result)
    
    def encode(self, x):
        scores, embeddings, spectrogram = self.encoder(x)
        return embeddings
    
    def decode(self, x):
        res = self.decoder(x)
        return res
    
    def resample(self, x, origsr=44100, targetsr=16000):
        sound_data = librosa.resample(x, orig_sr=origsr, target_sr=targetsr)
        return sound_data 
    
    def define_encoder(self, *args, **kwargs):
        input_layer = keras.layers.Input(shape=(), dtype=tf.float32)
        yamnet_model = hub.KerasLayer('https://tfhub.dev/google/yamnet/1', trainable=False)
        scores, embeddings, spectrogram = yamnet_model(input_layer)
        return keras.Model(inputs=[input_layer], outputs=[scores, embeddings, spectrogram], name='yamnet_encoder')
        
    
    def define_decoder(self, *args, **kwargs):
        input_layer = keras.layers.Input((1024, 1), dtype=tf.float32)
        x = keras.layers.Conv1DTranspose(512, kernel_size=1, strides=1, padding='valid')(input_layer)
        x = keras.layers.Conv1DTranspose(16, kernel_size=1, strides=1, padding='valid')(x)
        return keras.Model(inputs=[input_layer], outputs=[x], name='yamnet_decoder')
        

    def chunk_audio(self, x, chunk_size=10):
        print("Chunking audio data...")
        print("Chunk size: ", chunk_size)
        x = np.array_split(x, chunk_size)
        return x
    

if __name__ == '__main__':
    # Load Tone file data
    sample_tone_path = '/mnt/g/My Drive/Dev_work_sync/Python_dev/Python Projects/rock-tone-ml/data/Marshall Plexi - Dry.wav'
    label_tone_path = '/mnt/g/My Drive/Dev_work_sync/Python_dev/Python Projects/rock-tone-ml/data/Marshall Plexi - Amp.wav'

    sample_audio, sorig_sr = tsl.load_wav_data(sample_tone_path, length_sec=60, start=0)
    print("Sample audio: ", sample_audio)
    print("Sample audio shape: ", sample_audio.shape)
    print("Sample orig_sr: ", sorig_sr)

    label_audio, lorig_sr = tsl.load_wav_data(label_tone_path, length_sec=60, start=0)
    print("Sample audio: ", label_audio)
    print("Sample audio shape: ", label_audio.shape)
    print("Sample orig_sr: ", lorig_sr)
    
    # Initialize the YAMToneNet
    model = YAMToneNEt()

    # Resample the audio data
    sample_audio = model.resample(sample_audio, origsr=sorig_sr, targetsr=16000)
    label_audio = model.resample(label_audio, origsr=lorig_sr, targetsr=16000)
    
    # Chunk the audio data
    chunk = sample_audio.shape[0] / (0.96 * 16000)
    sample_audio = model.chunk_audio(sample_audio, chunk_size=chunk)
    print("Sample audio chunks: ", len(sample_audio))
    label_audio = model.chunk_audio(label_audio, chunk_size=chunk)

    # Load audio chunk embeddings into dataframe
    df = pd.DataFrame()
    df['sample_embeddings'] = [model.encode(x) for x in sample_audio]
    df['label_embeddings'] = [model.encode(x) for x in label_audio]
    print(" df ", df)

    df['predicted'] = [model.decode(x) for x in df['sample_embeddings']]
    print(" df ", df)

       


   



