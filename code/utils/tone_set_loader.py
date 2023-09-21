import sys
import matplotlib.pyplot as plt

sys.path.append('code\\')
import utils.audio_tools as audt
audt.test_import()

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