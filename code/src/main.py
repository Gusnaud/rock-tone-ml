import numpy as np
import sys
sys.path.append('code')
import utils.tone_set_loader as tsl
# audt.test_import()
# import tone_set_loader as tsl
import rock_tone_ml as ampnet
# from torchvis import make_dot
def main():
    # WAV file to load
    wav_file = 'data/Marshall Plexi - Dry.wav'
    target_wav_file = 'data/Marshall Plexi - Amp.wav'

    # Load samples from given wav file
    samples = tsl.load_wav_data(file_path=wav_file, length_mod=0.0001)
    # print(len(samples))
    # print(samples[0])

    # Target tone file
    target_samples = tsl.load_wav_data(file_path=target_wav_file, length_mod=0.0001)

    # Sample diffs
    diff_samples = [ss - tt for ss, tt in zip(samples, target_samples)]

    # # Plot and vizualise samples
    # tsl.plot_samples(samples=samples)
    # tsl.plot_samples(samples=target_samples)
    # tsl.plot_samples(samples=diff_samples)

    # Create new model
    # model = ampnet.ToneNet()
    model = ampnet.VocoderCNN()
    # model.double()
    print(model)

    # # Generate the visualization
    # dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    # dot.render(filename='model_graph', format='png', cleanup=True)

    # Create dataset
    train_ds = ampnet.WavDataset(samples, target_samples)

    ampnet.train_net(model=model, epochs=3, 
                     train_ds=train_ds, learn_rate=1e-4,
                     save_to_file=True
    )

    # Use the created model for inference on new signals
    model.eval()
    print(samples[:5])
    modulated_data = model(samples)
    print(modulated_data)
    # tsl.write_wav_file(file_path='data/Marshall_Plexi_modulated.wav', data=modulated_data)

    
if __name__ == '__main__':
    main()