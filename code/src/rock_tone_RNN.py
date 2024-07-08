'''
Following the structure of the rock_tone_ml.py definitions, this script is to independently test RNN based 
models for the purposes and research of this project. 
'''

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchmetrics.functional.regression import r2_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

### CLASSES #####
# Rock tone class definition
class ToneNet_RNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ToneNet_RNN, self).__init__() 
        self.num_layers = args[0][0]
        self.directions = 2 if args[0][1]>1 else 1
        self.hidden_size = args[0][1]
        self.RNN_net = nn.RNN(input_size=1, hidden_size=self.hidden_size, num_layers=args[0][0], nonlinearity='tanh', 
                              batch_first=True, bidirectional=True if self.directions > 1 else False, dropout=0.0
        )

    '''
    Sizing reference https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677
    Input:
        h0 size = [num_layers * num_directions, current_batch_size, input size] 
        input size = [seq_len, batch_size, input_size] - swap first two if batch_first=True
        input_size = number of features to use as input (columns in data)
    Output:
        output = [seq_len, batch_size, num_directions * hidden_size] - swap first two if batch_first=True
        h_n = [num_layers * num_directions, batch_size, hidden_size] - Unaffected by batch_first
        hidden_size = number of outputs in the last hidden state (ie number of time steps to predict).
    '''
    def forward(self, x, hn):
        res, hn = self.RNN_net(x, hn)
        return res, hn
    
    def get_h_first_dim(self):
        return self.num_layers * self.directions
    def get_h_out_size(self):
        return self.hidden_size


class WavDataset(Dataset):
    def __init__(self, data_samples, labels, seg_len=1, transforms=None):
        """
        Args:
            data_samples (list): List of data samples.
            labels (list): Corresponding labels for the data samples.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        # batch samples
        # seg_samples = [data_samples[ii:ii+batch_size] for ii in range(0, len(data_samples)-batch_size)]
        # seg_labels = [labels[ii:ii+batch_size] for ii in range(0, len(labels)-batch_size)]

        self.data_samples = data_samples#[data_samples[ii:ii+batch_size] for ii in range(0, len(data_samples)-batch_size)]
        self.labels = labels#[labels[ii:ii+batch_size] for ii in range(0, len(labels)-batch_size)]
        self.transforms = transforms
        self.seg_len = seg_len
    
    def __len__(self):
        """
        Returns the total number of samples in this dataset.
        """
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        """
        Retrieves the sample and its label at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            sample (Tensor): The data sample at the given index.
            label (Tensor): The data target at the given index.
        """
        if idx + self.seg_len >= self.data_samples.shape[0]:
            sample = torch.tensor(self.data_samples[self.data_samples.shape[0]-self.seg_len : self.data_samples.shape[0]], dtype=torch.float32)
            label = torch.tensor(self.labels[self.labels.shape[0]-self.seg_len : self.labels.shape[0]], dtype=torch.float32)
        else:
            sample = torch.tensor(self.data_samples[idx:idx + self.seg_len], dtype=torch.float32)
            label = torch.tensor(self.labels[idx:idx + self.seg_len], dtype=torch.float32)

        # print(torch.unsqueeze(sample, 0))
        # print(torch.unsqueeze(label, 0))
        
        # sample =self.data_samples[idx]
        # label = self.labels[idx]
        
        if self.transforms:
            sample = self.transforms(sample)
        
        # Ensure correct dimentionality when returning a batch
        # return sample.unsqueeze(0).unsqueeze(0), label.unsqueeze(0).unsqueeze(0)
        return sample.unsqueeze(-1), label.unsqueeze(-1)
        # return sample, label


### FUNCITONS #####
# Train the model passed as argument
def train_net(model=None, 
              epochs=30, 
              batch_size=64,
              loss_func=nn.MSELoss(),
              optimizer = None,
              learn_rate= 1e-4, 
              train_ds= None,
              val_ds= None,
              test_ds= None,
              save_to_file= False):
    
    # Set device
    if torch.cuda.is_available():
        print("CUDA device is Available -> ", torch.cuda.device_count())
    else:
        print("No CUDA device, using CPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate the network and send it to device
    model = model.to(device)
    # summary(model, input_size=[[1, 1],[10, 1]], batch_size=batch_size)

    plt.ion()
    graph = None
    loss_hist = list()
    epoch_lst = list()
    r2_hist = list()

    # Define a Loss function and optimizer
    criterion = loss_func
    if optimizer == None:
        print("I: Setting Default Adam optimizer, because none were given.")
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                            shuffle=False, num_workers=8
    )

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_R2 = 0.0
        steps = 0
        for data_i, data_l in tqdm(trainloader):
            # print("D:", data_i.shape)
            # print("D:", data_l.shape)

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data_i.to(device), data_l.to(device)
            # Depending on goal, re-init h0 for every forward pass (https://discuss.pytorch.org/t/how-to-handle-last-batch-in-lstm-hidden-state/40858)
            # Pytorch defaults to zeroes if h0 not given. 
            h0 = torch.randn(model.get_h_first_dim(), data_i.shape[0], model.get_h_out_size())
            h0 = h0.to(device)
            # print("h0 shape:", h0.shape)

            # forward + backward + optimize
            outputs, hn = model(inputs, h0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            steps += 1

            running_loss += outputs.shape[0] * loss.item()
            running_R2 += r2_score(inputs.flatten(), outputs.flatten())
            
            # zero the parameter gradients
            optimizer.zero_grad()

        # Print loss for each epoch    
        r2_hist.append(running_R2 / float(len(trainloader.dataset)))
        loss_hist.append(running_loss / float(len(trainloader.dataset)))
        epoch_lst.append(epoch + 1)
        print('[Epoch: %d, Steps: %5d, loss: %.3f, R2: %.6f]' % (epoch_lst[-1], steps, loss_hist[-1], r2_hist[-1]))

        if graph == None:
            # plotting the first frame
            graph = plt.plot(epoch_lst, loss_hist, 'b')[0]
            plt.xlabel("Epochs")
            plt.ylabel("Training Loss")
            plt.title("Training Loss History")
            plt.pause(1)

        graph.remove()
        graph = plt.plot(epoch_lst, loss_hist, 'b')[0]
        plt.pause(0.2)

    print('Finished Training')

    if save_to_file == True:
        print('Saving model...')
        torch.save(model, 'rock_tone_ml_model.pth')
        print('Model saved.')

    graph.figure.savefig('data/test.png')
    graph.figure.show()


    