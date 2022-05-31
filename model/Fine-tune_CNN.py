import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(1024, 1024, 3, padding=1)
        )
    def forward(self, x):
        return self.encoder(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(512,  128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(128,  32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32,  3, kernel_size=2, stride=2),
        )
 
    def forward(self, x):
        encode = self.encoder(x)
        print(encode)
        output = self.decoder(encode)
        return encode, output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = Encoder()
        self.fc = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 8),
#             nn.Softmax()
        )
 
    def forward(self, x):
        x = self.encoder(x)
#         x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x