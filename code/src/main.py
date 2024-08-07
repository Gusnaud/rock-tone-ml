import numpy as np
import sys
sys.path.append('code')
import utils.tone_set_loader as tsl
# audt.test_import()
# import tone_set_loader as tsl
import rock_tone_ml as ampnet
import rock_tone_RNN as ampnetRNN
# from torchvis import make_dot
def main():
    # WAV file to load
    wav_file = 'data/Marshall Plexi - Dry.wav'
    target_wav_file = 'data/Marshall Plexi - Amp.wav'
    
    train_flag = True
    start_secs = 0 * 60 + 30
    segment_length = 2
    audio_len_secs = 0 * 60 + 10
    train_epochs = 1
    batch_size = 512
    learn_rate = 1e-4
    is_segmented = True if segment_length > 1 else False

    rnn_num_layers = 2
    rnn_num_directions = 1
    rnn_hid_size = 1

    # Load samples from given wav file
    samples = tsl.load_wav_data(file_path=wav_file, length_sec=audio_len_secs, 
                                start=start_secs)

    # Target tone file
    target_samples = tsl.load_wav_data(file_path=target_wav_file, length_sec=audio_len_secs, 
                                       start=start_secs)

    # # Sample diffs
    # diff_samples = [ss - tt for ss, tt in zip(samples, target_samples)]

    # # Plot and vizualise samples
    # tsl.plot_samples(samples=samples, is_segmented=False)
    # tsl.plot_samples(samples=target_samples, is_segmented=False)
    # tsl.plot_samples(samples=diff_samples)

    import os.path
    if os.path.isfile('rock_tone_ml_model.pth') and not train_flag:
        model = ampnetRNN.torch.load('rock_tone_ml_model.pth')
    else:
        # Create new model
        # model = ampnet.ToneNet()
        # model = ampnet.VocoderCNN()
        # model = ampnet.ToneNet_NN()
        model = ampnetRNN.ToneNet_RNN([rnn_num_layers, rnn_num_directions, rnn_hid_size])
        # model.double()
        print(model)

        # Generate the visualization
        # dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
        # dot.render(filename='model_graph', format='png', cleanup=True)

        # Create dataset
        train_ds = ampnetRNN.WavDataset(samples, target_samples, seg_len=segment_length)

        ampnetRNN.train_net(model=model, epochs=train_epochs, 
                        train_ds=train_ds, learn_rate=learn_rate, batch_size=batch_size,
                        save_to_file=True
        )
    
    # Use the created model for inference on new signals
    model.eval()
    
    with ampnetRNN.torch.no_grad():
        # Set device
        if ampnetRNN.torch.cuda.is_available():
            print("CUDA device is Available -> ", ampnet.torch.cuda.device_count())
        else:
            print("No CUDA device, using CPU")
        device = ampnetRNN.torch.device("cuda:0" if ampnet.torch.cuda.is_available() else "cpu")
        samples_prep = ampnetRNN.torch.tensor(samples, dtype=ampnet.torch.float32)
        test_ds = ampnetRNN.WavDataset(samples_prep, target_samples, seg_len=segment_length)
        
        testloader = ampnetRNN.torch.utils.data.DataLoader(test_ds, batch_size=batch_size, 
                                            shuffle=False, num_workers=6
        )

        modulated_data = list()
        for data_i, data_l in ampnetRNN.tqdm(testloader):

            inputs = data_i.to(device)
            batch_sequence = list()
            for seq in model(inputs, None)[0].detach().cpu().numpy():
                batch_sequence.append(seq[0])
            modulated_data.append(np.asarray(batch_sequence))
        modulated_data = np.concatenate(modulated_data, axis=0)
        
        print("modulated_data shape from loader:", modulated_data.shape)
        modulated_data = modulated_data.flatten()
        print(modulated_data.shape)
        
        tsl.write_wav_file(file_path='data/Marshall_Plexi_modulated.wav', data=modulated_data)

    # Plot and vizualise samples
    tsl.plot_samples(samples=target_samples, title="modulated_data", is_segmented=False)
    
if __name__ == '__main__':
    main()