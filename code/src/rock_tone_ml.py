import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchmetrics.functional.regression import r2_score
# import torchvision
# import torchvision.transforms as transforms
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

### CLASSES #####
# Rock tone class definition
class ToneNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ToneNet, self).__init__(*args, **kwargs)
        self.fc_input = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding='same')
        self.fc_conv1 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=3, padding='same')
        self.fc_conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(self.fc_conv2.out_channels, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, 512)
        self.fc32 = nn.Linear(512, 512)
        self.drop = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, 32)
        self.fc_output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.fc_input(x))
        # x = torch.transpose(x, -1, -2)
        # print(x.shape)
        x = self.relu(self.fc_conv1(x))
        # print(x.shape)
        x = self.relu(self.fc_conv2(x))
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.relu(self.fc1(x))
        # print(x.shape)
        x = self.relu(self.fc2(x))
        # print(x.shape)
        x = self.relu(self.fc3(x))
        # print(x.shape)
        x = self.relu(self.fc31(x))
        # print(x.shape)
        x = self.relu(self.fc32(x))
        # print(x.shape)

        x = self.drop(x)
        # print(x.shape)
        x = self.relu(self.fc4(x))
        # print(x.shape)
        x = self.tan(self.fc_output(x))
        # print(x.shape)
        return x
    
class ToneNet_NN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ToneNet_NN, self).__init__(*args, **kwargs)
        self.full_net = nn.Sequential(
            # nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            
            # nn.Flatten(),

            # nn.Linear(1, 16),
            # nn.ReLU(),
            # nn.Linear(16, 32),
            # nn.ReLU(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            nn.Linear(1, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            # nn.Linear(512, 1024),
            # nn.ReLU(),

            nn.Dropout(p=0.5),

            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.full_net(x)


class VocoderCNN(nn.Module):
    def __init__(self):
        super(VocoderCNN, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),

            nn.Flatten(),

            nn.Dropout(p=0.5),

            nn.Linear(1024, 4096, True),
            nn.ReLU(),
            nn.Linear(4096, 4096, True),
            nn.ReLU(),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(4096, 4096, True),
            nn.ReLU(),
            nn.Linear(4096, 1024, True),
            nn.ReLU(),

            nn.Unflatten(1, (1024, 1)),

            # nn.ConvTranspose1d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),  # Tanh to ensure the output is within a range suitable for audio signals
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class WavDataset(Dataset):
    def __init__(self, data_samples, labels, transforms=None, batch_size=512):
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
        sample = torch.tensor(self.data_samples[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        # print(torch.unsqueeze(sample, 0))
        # print(torch.unsqueeze(label, 0))
        
        # sample =self.data_samples[idx]
        # label = self.labels[idx]
        
        if self.transforms:
            sample = self.transforms(sample)
        
        # Ensure correct dimentionality when returning a batch
        # return sample.unsqueeze(0).unsqueeze(0), label.unsqueeze(0).unsqueeze(0)
        return sample.unsqueeze(0), label.unsqueeze(0)


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
    # summary(model, input_size= (1, 1), batch_size=batch_size)
    # return

    plt.ion()
    graph = None
    loss_hist = list()
    r2_hist = list()
    epoch_lst = list()

    # Define a Loss function and optimizer
    criterion = loss_func
    if optimizer == None:
        print("I: Setting Default Adam optimizer, because none were given.")
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                            shuffle=True, num_workers=8
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

            # forward + backward + optimize
            outputs = model(inputs)
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


    