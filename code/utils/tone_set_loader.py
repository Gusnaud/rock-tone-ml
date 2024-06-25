import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('code\\')
import utils.audio_tools as audt
# audt.test_import()

def load_wav_data(file_path=None, length_sec=0.1, start=0, segment_l=512):
    if file_path == None:
        print("E: No file_path given. Exiting...")
        return None
    
    samples, sample_rate = load_audio_samples(file_path)
    samples = np.asarray(samples)
    print("sample rate: ", sample_rate)

    slice_len = int(length_sec * sample_rate)

    print('slice_len:', slice_len)
    start_sample_count = start * sample_rate
    samples = samples[start_sample_count:start_sample_count + slice_len]
    
    # Norm between -1 and 1
    samples = samples / np.max(samples)

    # # Slice the samples list into segments of size L
    # seg_samples = [samples[ii:ii+segment_l] for ii in range(0, len(samples)-segment_l)]
    # result = list()
    # for ll in seg_samples:
    #     for el in ll:
    #         result.append(el)
    return samples

def write_wav_file(file_path=None, data=None):
    audt.save_wav(file_path=file_path, data=data)
    return


def load_audio_samples(file_path=None):
    wave_obj = audt.read_wav(file_path, mode='rb')
    sample_rate = wave_obj.getframerate()
    samples = audt.decode_wav(wave_obj)
    return samples, sample_rate


def plot_samples(samples, title='Audio samples over time', is_segmented=True):
    plt_samples = samples.copy()
    if is_segmented:
        plt_samples = [ii for seg in samples for ii in seg]

    # x data for plotting
    x = [ii for ii in range(len(plt_samples))]
    print("Length of audio segment [Seconds]: ", len(plt_samples) / 44100)

    plt_samples = np.asarray(plt_samples)

    plt.figure(1)
    # if len(plt_samples.shape) > 1:
    #     for p in range(plt_samples.shape):
    #         plt.plot(x, plt_samples[p])
    # else:
    #     plt.plot(x, plt_samples)
    plt.plot(x, plt_samples)

    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('Sample')
    plt.grid(True)

    plt.pause(10)
    plt.savefig('data/{}.png'.format(title))
     