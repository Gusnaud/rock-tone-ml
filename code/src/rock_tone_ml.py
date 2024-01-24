import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

### CLASSES #####
# Rock tone class definition
class ToneNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ToneNet, self).__init__(*args, **kwargs)
        self.fc_input = nn.Linear(1, 512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc31 = nn.Linear(2048, 2048)
        self.fc32 = nn.Linear(2048, 2048)
        self.drop = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(2048, 512)
        self.fc_output = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc_input(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc31(x))
        x = self.relu(self.fc32(x))

        x = self.drop(x)
        x = self.relu(self.fc4(x))
        x = self.tan(self.fc_output(x))
        return x


class WavDataset(Dataset):
    def __init__(self, data_samples, labels, transforms=None):
        """
        Args:
            data_samples (list): List of data samples.
            labels (list): Corresponding labels for the data samples.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_samples = data_samples
        self.labels = labels
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
        
        if self.transforms:
            sample = self.transforms(sample)
        
        return torch.unsqueeze(sample, 0), torch.unsqueeze(label, 0)
        # return sample, label


### FUNCITONS #####
# Train the model passed as argument
def train_net(model=None, 
              epochs=10, 
              loss_func=nn.MSELoss(),
              optimizer = None,
              learn_rate= 0.01, 
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

    # Define a Loss function and optimizer
    criterion = loss_func
    if optimizer == None:
        print("I: Setting Default Adam optimizer, because none were given.")
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=64, 
                                            shuffle=True, num_workers=1
    )

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        steps = 0
        for data_i, data_l in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data_i.to(device), data_l.to(device)
            # print(inputs.shape)
            # print(inputs[0], labels[0])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            steps += 1

            running_loss += outputs.shape[0] * loss.item()
            # print(loss)

        # Print loss for each epoch    
        print('[Epoch: %d, Steps: %5d, loss: %.3f]' % (epoch + 1, steps, running_loss / float(len(trainloader.dataset))))
                

    print('Finished Training')

    if save_to_file == True:
        torch.save(model, 'rock_tone_ml_model.pth')


    