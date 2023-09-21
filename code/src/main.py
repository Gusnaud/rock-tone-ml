import numpy as np
sys.path.append('code\\')
import utils.tone_set_loader as tsl
# audt.test_import()
# import tone_set_loader as tsl
import rock_tone_ml as ampnet


def load_wav_data(file_path=None, length_mod=0.002):
    if file_path == None:
        print("E: No file_path given. Exiting...")
        return None
    
    samples = tsl.load_audio_samples(file_path)
    slice_len = int(len(samples) * length_mod)
    samples = samples[:slice_len]
    samples = samples / np.max(samples)
    return samples[:slice_len]


def main():
    # WAV file to load
    wav_file = 'data\Marshall Plexi - Dry.wav'
    target_wav_file = 'data\Marshall Plexi - Amp.wav'
    # Load samples from given wav file
    samples = load_wav_data(file_path=wav_file, length_mod=0.0001)
    # Target tone file
    target_samples = load_wav_data(file_path=target_wav_file, length_mod=0.0001)
    # Sample diffs
    diff_samples = samples - target_samples
    # Plot and vizualise samples
    tsl.plot_samples(samples=samples)
    tsl.plot_samples(samples=target_samples)
    tsl.plot_samples(samples=diff_samples)

    # Create new model
    


if __name__ == '__main__':
    main()