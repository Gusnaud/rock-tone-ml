import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

### CLASSES #####
# Rock tone class definition
class ToneNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ToneNet, self).__init__(*args, **kwargs)
        self.fc_input = nn.Linear(128, 1024)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc_output = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc_input(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tan(self.fc_output(x))


# Dataset class for audio data. 
# Labels are desired output samples.
# class WavDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

### FUNCITONS #####
# Train the model passed as argument
def train_net(model=None, 
              epochs=10, 
              loss_func=nn.MSELoss(),
              optimizer = None,
              learn_rate= 0.001, 
              train_ds= None,
              val_ds= None,
              test_ds= None):
    
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
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # # Load CIFAR-10 data
    # transform = transforms.Compose([transforms.ToTensor(), 
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
    #                                         download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, 
                                            shuffle=True, num_workers=6)

    # Train the network
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training')