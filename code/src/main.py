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
    
    start_secs = 4 * 60 + 30
    segment_length = 1
    audio_len_secs = 5
    train_epochs = 20
    batch_size = 512
    learn_rate = 1e-3
    is_segmented = True if segment_length > 1 else False

    # Load samples from given wav file
    samples = tsl.load_wav_data(file_path=wav_file, length_sec=audio_len_secs, 
                                start=start_secs, segment_l=segment_length)

    # Target tone file
    target_samples = tsl.load_wav_data(file_path=target_wav_file, length_sec=audio_len_secs, 
                                       start=start_secs, segment_l=segment_length)

    # Sample diffs
    diff_samples = [ss - tt for ss, tt in zip(samples, target_samples)]

    # # Plot and vizualise samples
    tsl.plot_samples(samples=samples, is_segmented=False)
    tsl.plot_samples(samples=target_samples, is_segmented=False)
    # tsl.plot_samples(samples=diff_samples)

    # Create new model
    # model = ampnet.ToneNet()
    model = ampnet.VocoderCNN()
    # model.double()
    print(model)

    # Generate the visualization
    # dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    # dot.render(filename='model_graph', format='png', cleanup=True)


    # Create dataset
    train_ds = ampnet.WavDataset(samples, target_samples, batch_size=segment_length)

    ampnet.train_net(model=model, epochs=train_epochs, 
                     train_ds=train_ds, learn_rate=learn_rate, batch_size=batch_size,
                     save_to_file=True
    )

    # Use the created model for inference on new signals
    model.eval()
    
    with ampnet.torch.no_grad():
        # Set device
        if ampnet.torch.cuda.is_available():
            print("CUDA device is Available -> ", ampnet.torch.cuda.device_count())
        else:
            print("No CUDA device, using CPU")
        device = ampnet.torch.device("cuda:0" if ampnet.torch.cuda.is_available() else "cpu")
        samples_prep = ampnet.torch.tensor(samples, dtype=ampnet.torch.float32)
        samples_prep = samples_prep.unsqueeze(0).unsqueeze(0)
        print("samples shape from loader:", samples_prep.shape)
        inputs = samples_prep.to(device)
        modulated_data = model(inputs)
        print(modulated_data)
        print("modulated_data shape from loader:", modulated_data.shape)
        modulated_data = modulated_data.detach().cpu().numpy()
        modulated_data = modulated_data.flatten()
        print(modulated_data.shape)
        

        tsl.write_wav_file(file_path='data/Marshall_Plexi_modulated.wav', data=modulated_data)

    # Plot and vizualise samples
    tsl.plot_samples(samples=target_samples, title="modulated_data", is_segmented=False)
    # tsl.plot_samples(samples=target_samples, title="target_samples")

    
if __name__ == '__main__':
    main()