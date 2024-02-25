import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('code\\')
import utils.audio_tools as audt
# audt.test_import()

def load_wav_data(file_path=None, length_mod=0.02):
    if file_path == None:
        print("E: No file_path given. Exiting...")
        return None
    
    samples = np.asarray(load_audio_samples(file_path))
    slice_len = int(len(samples) * length_mod)
    print('slice_len:', slice_len)
    samples = samples[int(len(samples)/2):int(len(samples)/2) + slice_len]
    samples = samples / np.max(samples)
    print(samples)
    # Slice the samples list into segments of size L
    seg_samples = [samples[ii:ii+512] for ii in range(0, len(samples)-512)]
    return seg_samples


def load_audio_samples(file_path=None):
    wave_obj = audt.read_wav(file_path, mode='rb') 
    samples = audt.decode_wav(wave_obj)
    return samples


def plot_samples(samples):
    # x data for plotting
    x = [ii for ii in range(len(samples))]
    print("Length of audio file [Samples]: ", len(samples) / 44100)

    plt.figure(1)
    plt.plot(x, samples)
    plt.title('Audio samples over time')
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    plt.grid(True)
    plt.show() 