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

# 1. Input layer
# 2. Convolutional layer
# 3. ReLU layer
# 4. Pooling layer
# 5. Fully-connected layer
class Galaxy(nn.Module):
  def __init__(self):
    super(Galaxy, self).__init__()
    # input_shape = (3, 224, 224)
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
    #                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    # in_channels: Input channels
    # out_channels: Output channels
    # kernel_size: Fillter size
    # stride: Each step our Fillter move
    # padding: We want our image can remain origin size
    self.cnn = nn.Sequential(
        ## CNN1
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1), # padding = kernel_size / 2
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (64, 112, 112)
        ## CNN2
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (128, 56, 56)
        ## CNN3
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (256, 28, 28)
        ## CNN4
        nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (512, 14, 14)
        ## CNN5
        nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2)## (512, 7, 7)
        ## CNN6
#         nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size = 2)## (1024, 3, 3)
    )
    self.fc = nn.Sequential(
        nn.Linear(512 * 7 * 7, 256), # Fully-connected layer
        nn.Dropout(0.5), # Avoid overfitting
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.Dropout(0.5),
        nn.ReLU(),
#         nn.Linear(512, 256),
#         nn.Dropout(0.5),
#         nn.ReLU(),
        nn.Linear(128, 3)
    )
  # forward propagation
  def forward(self, x):
    x = self.cnn(x)
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

    train_image_path = "/home/chisc/workspace/wuzhenrong/galaxy/three_final/train/"
    val_image_path = "/home/chisc/workspace/wuzhenrong/galaxy/three_final/validation/"
    test_image_path = "/home/chisc/workspace/wuzhenrong/galaxy/three_final/test/"

    batch_size = 32
    # torch.manual_seed(0) # random seed

    # transforms.GaussianBlur(7,3)
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # class0 = 212
    # class1 = 158
    # class2 = 2209
    # class_weights = [class0/2209, class1/2209, class2/2209]

    # train_set_size = int(len(train_data) * 0.9)
    # valid_set_size = len(train_data) - train_set_size
    # train_set, val_set = torch.utils.data.random_split(train_data, [train_set_size, valid_set_size])
    # print(train_data)

    # class0 = 351
    # class1 = 265
    # class2 = 1839
    # class_weights = [class0/2455, class1/2455, class2/2455]
    # target = torch.cat((torch.zeros(class0), torch.ones(class1), torch.ones(class2)*2)).long()
    # weights = 1. / torch.tensor(class_weights, dtype=torch.float)
    # # weights = weights.double()
    # # print(weights)
    # target = target[torch.randperm(len(target))]# shuffle
    # sampler_weights = weights[target]
    # class_sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=2000, replacement=True)



    # E type :212
    # I type :158
    # S type :1839
    # All :2209
    ### new
    # E type :351
    # I type :265
    # S type :1839
    # All :2455

    ### latest
    # E type :732
    # I type :270
    # S type :1466
    ## All :2468
    train_trans = transforms.Compose([
    #                                   transforms.RandomHorizontalFlip(),
    #                                   transforms.RandomRotation((-30, 30)),
    #                                   transforms.Resize((255, 255)),
    #                                   transforms.CenterCrop(210),
                                    transforms.Resize((255, 255)),
    #                                   transforms.GaussianBlur(7,3),
    #                                   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.ToTensor()])

    train_data = ImageFolder(train_image_path, transform = train_trans)
    train_loader = DataLoader(train_data, pin_memory = True, batch_size = batch_size, sampler=ImbalancedDatasetSampler(train_data))

    val_trans = transforms.Compose([transforms.Resize((255, 255)),transforms.ToTensor()])
    val_data = ImageFolder(val_image_path, transform = val_trans)
    val_loader = DataLoader(val_data, shuffle = True)

    test_trans = transforms.Compose([transforms.Resize((255, 255)),transforms.ToTensor()])
    test_data = ImageFolder(test_image_path, transform = test_trans)
    test_loader = DataLoader(test_data, shuffle = True)

    print('Train:', len(train_data))
    print('Valid:', len(val_data))
    print('Test:', len(test_data))

    # for i, j in train_loader:
    #     break
    images, labels = next(iter(train_loader))
    print(labels)
    for i in np.arange(3):
        print(labels[i])
        plt.figure(i)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.show()

    model = Galaxy()
    model = model.to(device)
    print(model)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0008)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.01)
    loss_func = nn.CrossEntropyLoss()
    # # reg = JacobianReg() # Jacobian regularization
    # lambda_JR = 0.01 # hyperparameter
    l1_crit = nn.L1Loss(size_average=False)
    factor = 0.005
    n_epochs = 40
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
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            reg_loss = 0
            reg_loss = l1_crit(prediction.argmax(dim = 1), y)
            
            super_loss = loss_func(prediction, y)
            
            loss = super_loss + factor*reg_loss
            
            loss.backward()
            optimizer.step()
            
            acc = ((prediction.argmax(dim = 1) == y).float().mean())
            train_acc += acc/len(train_loader)
            train_loss += loss/len(train_loader)
        print(f"[ Train | {epoch+1}/{n_epochs} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        train_loss_record.append(train_loss)
        train_acc_record.append(train_acc)

        model.eval()
        for x, y in tqdm(val_loader):
            x, y = x.to(device), y.to(device)
            prediction = model(x)
    #         reg_loss = 0

    #         reg_loss = l1_crit(prediction.argmax(dim = 1), y)
            
            loss = loss_func(prediction, y)
            
    #         loss = super_loss + factor*reg_loss
            
            loss.backward()
            acc = ((prediction.argmax(dim = 1) == y).float().mean())
            val_acc += acc/len(val_loader)
            val_loss += loss/len(val_loader)
            if loss < min_loss:
                min_loss = loss
                torch.save(model, 'E_I_S_new.pkl')
        print(f"[ Validation | {epoch+1}/{n_epochs} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")
        val_loss_record.append(val_loss)
        val_acc_record.append(val_acc)
    # torch.save(model, 'E_I_Sc.pkl')
    
    actu = []
    ai_pred = []
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        actu.append(y.to('cpu').numpy()[0])
        ai_pred.append(prediction.argmax().to('cpu').numpy().tolist())
        loss = loss_func(prediction, y)
        loss.backward()
        acc = ((prediction.argmax(dim = 1) == y).float().mean())
        test_acc += acc/len(test_loader)
        test_loss += loss/len(test_loader)
    print(f"[ Test ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")


    model = torch.load('E_I_S_new.pkl')
    loss_func = nn.CrossEntropyLoss()
    i = 0
    for x, y in test_loader:
        i += 1
        if train_on_gpu:
            x, y = x.cuda(), y.cuda()
        output = model(x)
        out = output.argmax(dim = 1)
        out = out.to('cpu').numpy()
        if y[0] == 0:
            ans = 'E'
        elif y[0] == 1:
            ans = 'I'
        else:
            ans = 'S'
        if i % 10 == 0:
            plt.figure(i)
            if out[0] == 0:
                plt.title(f'Predict: E, Answer: {ans}')
            elif out[0] == 1:
                plt.title(f'Predict: I, Answer: {ans}')
            else:
                plt.title(f'Predict: S, Answer: {ans}')
            x = x.squeeze()
            x = x.to('cpu')
            plt.imshow(x.permute(1, 2, 0))
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

    answer = pd.Series(actu, name='Actual')
    pred = pd.Series(ai_pred, name='Predicted', dtype = 'int64')
    df_confusion = pd.crosstab(answer, pred, rownames=['True'], colnames=['Predicted'], margins=True)




if __name__ == '__main__':
    train()