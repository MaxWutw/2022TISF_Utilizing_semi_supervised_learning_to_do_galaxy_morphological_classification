import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import DatasetFolder
import torchvision
from tqdm.notebook import tqdm as tqdm
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import copy
import os
from torchsummary import summary
from sklearn.metrics import f1_score

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
#             nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
#             nn.BatchNorm1d(256),
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


def train():
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available!')
    device = "cuda" if train_on_gpu else "cpu"
    # device = 'cpu'
    RESUME = True

    # supervised_path = "/home/max/Desktop/galaxy_science_display/eight_type/train/"
    # val_image_path = "/home/max/Desktop/galaxy_science_display/galaxy_image/eight_type/validation/"
    # test_image_path = "/home/max/Desktop/galaxy_science_display/galaxy_image/eight_type/test/"
    unsupervised_path = "/home/max/Desktop/galaxy_science_display/images_gz2/"

    supervised_path = "/home/max/Desktop/galaxy_science_display/eight_type/train/"
    # supervised_path = "/home/max/Desktop/galaxy_science_display/galaxy_image/eight_type/train"
    val_image_path = "/home/max/Desktop/galaxy_science_display/eight_type/validation/"
    test_image_path = "/home/max/Desktop/galaxy_science_display/eight_type/test/"


    batch_size = 32
    train_trans = transforms.Compose([
    #                                   transforms.RandomHorizontalFlip(),
    #                                   transforms.RandomRotation((-30, 30)),
    #                                   transforms.Resize((256, 256)),
    #                                   transforms.RandomCrop(size=(100, 100)),
                                    transforms.Resize((256, 256)),
    #                                   transforms.CenterCrop(200),
    #                                   transforms.Resize((256, 256)),
    #                                   transforms.Resize((255, 255)),
    #                                   transforms.GaussianBlur(7,3),
    #                                   transforms.ColorJitter(contrast=0.8),
                                    transforms.ToTensor()])
    train_data = ImageFolder(supervised_path, transform=train_trans)
    train_loader = DataLoader(train_data, pin_memory=True, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_data))

    val_trans = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
    val_data = ImageFolder(val_image_path, transform = val_trans)
    val_loader = DataLoader(val_data, shuffle = True)

    test_trans = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
    test_data = ImageFolder(test_image_path, transform = test_trans)
    test_loader = DataLoader(test_data, shuffle = True)

    unsuper_trans = transforms.Compose([
    #                                   transforms.RandomHorizontalFlip(),
    #                                   transforms.RandomRotation((-30, 30)),
                                    transforms.Resize((256, 256)),
    #                                   transforms.RandomCrop(size=(200, 200)),
    #                                   transforms.Resize((256, 256)),
    #                                   transforms.CenterCrop(200),
    #                                   transforms.Resize((256, 256)),
    #                                   transforms.GaussianBlur(7,3),
    #                                   transforms.ColorJitter(contrast=0.8),
    #                                   transforms.RandomResizedCrop((224, 224)),
                                    transforms.ToTensor()])
    unsuper_data = ImageFolder(unsupervised_path, transform=unsuper_trans)
    unsuper_loader = DataLoader(unsuper_data, pin_memory=True, batch_size=batch_size, shuffle=True)


    images, labels = next(iter(train_loader))
    print(labels)
    for i in np.arange(3):
        print(labels[i])
        plt.figure(i)
        plt.title('EFIGI')
        plt.imshow(images[i].permute(1, 2, 0))
        plt.show()
        
    un_images, un_labels = next(iter(unsuper_loader))
    print(un_labels)
    for i in np.arange(3):
        print(un_labels[i])
        plt.figure(i)
        plt.title('GZ2')
        plt.imshow(un_images[i].permute(1, 2, 0))
        plt.show()



    print(summary(CNN().to('cuda'), (3, 256, 256)))

    pre_model = torch.load('autoencoder_pretrain_new_version_EfFIGI_GZ2.pkl')
    finetune_model = CNN().to(device) # 14
    # print(finetune_model.state_dict())
    data = finetune_model.state_dict()
    # print(data.keys())
    j = 0


    for i in finetune_model.parameters():
        if j == 14:
            break
        j += 1
        i.requires_grad=False

    # for i in finetune_model.parameters():
    #     print(i)
    epoch = 1
    finetune_model.encoder.load_state_dict(copy.deepcopy(pre_model.encoder.state_dict())) # loads encoder weights from pretrained model
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=3e-4)
    optimizer = torch.optim.Adam(finetune_model.parameters(), lr=3e-4, weight_decay=0.01)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=0.00001)
    loss_func = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(finetune_model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.01)
    # loss_func = nn.CrossEntropyLoss()
    # state = {"model":finetune_model.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch}
    # if not os.path.isdir("checkpoint"):
    #     os.mkdir("checkpoint")
    # if RESUME:
    #     checkpoint_path = "./checkpoint/best_4.pth"
    #     checkpoint = torch.load(checkpoint_path)
    #     finetune_model.load_state_dict(checkpoint["model"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     epoch = checkpoint["epoch"]
    #     print(epoch)

    # # reg = JacobianReg() # Jacobian regularization
    # lambda_JR = 0.01 # hyperparameter
    # l1_crit = nn.L1Loss(size_average=False)
    # factor = 0.03
    n_epochs = 50
    train_loss_record = []
    train_acc_record = []
    val_loss_record = []
    val_acc_record = []
    min_loss = 2000.
    for epoch in range(n_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        finetune_model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = finetune_model(x)
    #         reg_loss = 0
    #         reg_loss = l1_crit(prediction.argmax(dim = 1), y)
            
            loss = loss_func(prediction, y)
            
    #         loss = super_loss + factor*reg_loss
            
            loss.backward()
            optimizer.step()
            
            acc = ((prediction.argmax(dim = 1) == y).float().mean())
            train_acc += acc/len(train_loader)
            train_loss += loss/len(train_loader)
        print(f"[ Train | {epoch+1}/{n_epochs} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        train_loss_record.append(train_loss.item())
        train_acc_record.append(train_acc.item())

        finetune_model.eval()
        for x, y in tqdm(val_loader):
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                prediction = finetune_model(x)
    #             reg_loss = 0

    #             reg_loss = l1_crit(prediction.argmax(dim = 1), y)

                loss = loss_func(prediction, y)

    #             loss = super_loss + factor*reg_loss

        #         loss.backward()
                acc = ((prediction.argmax(dim = 1) == y).float().mean())
                val_acc += acc/len(val_loader)
                val_loss += loss/len(val_loader)
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(finetune_model, "finetune_model_new_version456.pkl")
            print("Model Saved")
    #             torch.save(state, "./checkpoint/best_%s.pth" % (str(epoch)))
    #             torch.save(finetune_model, 'E_I_S_new.pkl')
        print(f"[ Validation | {epoch+1}/{n_epochs} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")
        val_loss_record.append(val_loss.item())
        val_acc_record.append(val_acc.item())
    # torch.save(model, 'E_I_Sc.pkl')

    # model_fine = torch.load('finetune_model_new_version456.pkl')
    actu = []
    ai_pred = []
    finetune_model.eval()
    test_acc = 0.0
    test_loss = 0.0
    for x, y in tqdm(test_loader):
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            prediction = finetune_model(x)
            actu.append(y.to('cpu').numpy()[0])
            ai_pred.append(prediction.argmax().to('cpu').numpy().tolist())
            loss = loss_func(prediction, y)
        #     loss.backward()
            acc = ((prediction.argmax(dim = 1) == y).float().mean())
            test_acc += acc/len(test_loader)
            test_loss += loss/len(test_loader)
    print(f"[ Test ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")

    model = torch.load('finetune_model_new_version456.pkl')
    galaxy_type = ['E', 'S0', 'Sa', 'Sb', 'SBa', 'SBb', 'SBc', 'Sc']
    loss_func = nn.CrossEntropyLoss()
    i = 0
    data = []
    for x, y in val_loader:
        i += 1
        if train_on_gpu:
            x, y = x.cuda(), y.cuda()
        output = model(x)/10
        print(output)
        out = output.argmax(dim = 1)
    #     out = out.to('cpu').numpy()
        ans = galaxy_type[y[0]]
        pre = galaxy_type[out[0]]
        if ans == pre:
            data.append(output[0][out].item())
            print(output[0][out].item())
            plt.figure(i)
            plt.title(f'Predict: {pre}, Answer: {ans}')
    #         if out[0] == 0:

    #         elif out[0] == 1:
    #             plt.title(f'Predict: {pre}, Answer: {ans}')
    #         else:
    #             plt.title(f'Predict: S, Answer: {ans}')
            x = x.squeeze()
            x = x.to('cpu')
            plt.imshow(x.permute(1, 2, 0))
            plt.show()
            
            
    #     if i % 10 == 0:
            
    #         plt.figure(i)
    #         plt.title(f'Predict: {pre}, Answer: {ans}')
    # #         if out[0] == 0:
                
    # #         elif out[0] == 1:
    # #             plt.title(f'Predict: {pre}, Answer: {ans}')
    # #         else:
    # #             plt.title(f'Predict: S, Answer: {ans}')
    #         x = x.squeeze()
    #         x = x.to('cpu')
    # #         plt.imshow(x.permute(1, 2, 0))
    # #         plt.show()

    a = []
    for i in range(len(data)):
        a.append(i)
    plt.scatter(a, data)
    plt.show()

    plt.figure(1)
    plt.title('Training and Validation Loss')
    train_l, = plt.plot(train_loss_record, color = 'red')
    val_l, = plt.plot(val_loss_record, color = 'blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(handles = [train_l, val_l], labels = ['Training', 'Validation'], loc = 'best')
    plt.show()

    plt.figure(2)
    plt.title('Training and Validation Accuracy')
    train_a, = plt.plot(train_acc_record, color = 'red')
    val_a, = plt.plot(val_acc_record, color = 'blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(handles = [train_a, val_a], labels = ['Training', 'Validation'], loc = 'best')
    plt.show()

    confmat = confusion_matrix(y_true=actu, y_pred=ai_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],['E', 'S0', 'Sa', 'Sb', 'SBa', 'SBb', 'SBc', 'Sc'])
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7],['E', 'S0', 'Sa', 'Sb', 'SBa', 'SBb', 'SBc', 'Sc'])
    plt.show()

    f1score = f1_score(actu, ai_pred, average='weighted')
    print(f1score)

if __name__ == '__main__':
    train()