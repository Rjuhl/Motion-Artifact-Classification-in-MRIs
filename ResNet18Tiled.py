import nibabel as nib
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import random

STATUS = True
fold = 0
epochs = 20
batch_size = 256 if STATUS else 60
g_mean = 443.8712
g_std = 743.1469

BASE_DIR = "/Users/rainjuhl/PycharmProjects/MRI_BC_Project/New-Dataset"
SAVE_DIR = ""
IMG_DIR = ""

if STATUS:
    BASE_DIR = "/scratch/users/rjuhl/New-Dataset"
    SAVE_DIR = "/home/users/rjuhl/MRI_BC/ResNet2Res-Tiled"
    IMG_DIR = "/scratch/users/rjuhl/imgpics"


def getPatientID(file):
    if file[0] == 'N' or file[0] == 'R':
        return file[file.find('_') + 2: file.find('_') + 7]
    return file[file.find('-') + 1: file.find('-') + 6]


def getWeight():
    a = 0
    c = 0
    for file in os.listdir(BASE_DIR):
        if file.startswith('.'):
            continue
        elif file[0] == 'N':
            c += 1
        else:
            a += 1
    return c / a


def getFiles():
    patScanDict = {}
    fc = 0
    for file in os.listdir(BASE_DIR):
        if file.startswith('.'):
            continue
        fc += 1
        id = getPatientID(file)
        if id in patScanDict:
            patScanDict[id].append(file)
        else:
            patScanDict[id] = []
            patScanDict[id].append(file)
    return patScanDict


def getType(file):
        if file[0] == 'N' or file[0] == 'R':
            return file[file.find('.') - 1]
        return file[0]




def getQuotas(fileDict):
    aa, ac, ba, bc, ca, cc, da, dc, ea, ec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for key in fileDict:
        for file in fileDict[key]:
            cite = getType(file)
            type = False if file[0] == 'N' else True
            if cite == 'A':
                if type:
                    aa += 1
                else:
                    ac += 1
            if cite == 'B':
                if type:
                    ba += 1
                else:
                    bc += 1
            if cite == 'C':
                if type:
                    ca += 1
                else:
                    cc += 1
            if cite == 'D':
                if type:
                    da += 1
                else:
                    dc += 1
            if cite == 'E':
                if type:
                    ea += 1
                else:
                    ec += 1
    return [round(aa * 0.2), round(ac * 0.2), round(ba * 0.2), round(bc * 0.2),
            round(ca * 0.2), round(cc * 0.2), round(da * 0.2), round(dc * 0.2),
            round(ea * 0.2), round(ec * 0.2)]


def getCompatibility(files, aq, cq):
    ac = 0
    cc = 0
    for file in files:
        if file[0] == 'N':
            cc += 1
        else:
            ac += 1

    com = True if ac * aq >= 0 and cc * cq >= 0 else False

    return com, ac, cc


def getDataset():
    patients = getFiles()
    qutoas = getQuotas(patients)
    test = []
    train = []
    clean_quota = {'A': qutoas[1], 'B': qutoas[3], 'C': qutoas[5], 'D': qutoas[7], 'E': qutoas[9]}
    art_quota = {'A': qutoas[0], 'B': qutoas[2], 'C': qutoas[4], 'D': qutoas[6], 'E': qutoas[8]}
    print(f'{clean_quota}\n{art_quota}')
    for id in patients:
        files = patients[id]
        type = getType(files[0])
        com, ac, cc = getCompatibility(files, art_quota[type], clean_quota[type])
        if com:
            for file in files:
                test.append(file)
            clean_quota[type] -= cc
            art_quota[type] -= ac
        else:
            for file in files:
                train.append(file)

    return test, train


def fileWriter(fold, epoch, train_pre, test_pre, train_acc, test_acc, lost_arr):
    file = open(os.path.join(SAVE_DIR, "Fold#" + str(fold) + "Epoch#" + str(epoch) + "Prediction_and_acc_scores.txt"), 'a')
    file.write(str(fold) + str(epoch) + '\n')
    file.write("-----Training predictions-----\n")
    for t_file in train_pre:
        file.write(str(t_file) + ':' + str(train_pre[t_file]) + '\n')
    file.write("-----Testing predictions-----\n")
    for t_file in test_pre:
        file.write(str(t_file) + ':' + str(test_pre[t_file]) + '\n')
    file.write("-----Training and testing accuracies-----\n")
    file.write("Train Acc:" + str(train_acc) + '\n')
    file.write("Test Acc:" + str(test_acc) + '\n')
    file.write("Loss Arr:" + str(lost_arr) + '\n')
    file.write("----- END -----\n")
    file.close()


def saveResults(results):
    file = open('resultsFile.txt', 'a')
    file.write('START\n')
    for result in results:
        file.write(str(result) + '\n')
    file.write('END\n')
    file.close()


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def trans(img_data):
    size = img_data.size()[2]
    if size != 146:
        img_data = img_data[:, :, ((size - 146) // 2):size - ((size - 146) - ((size - 146) // 2))]
    return img_data


def normData(img_arr):
    return torch.div(torch.sub(img_arr, g_mean), g_std)


def basicNorm(i):
    return (i - torch.min(i)) / (torch.max(i) - torch.min(i))


class MRIDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        tile = np.load(os.path.join(BASE_DIR, self.files[index]))
        tile = torch.from_numpy(tile)
        tile = basicNorm(tile)
        val = 0 if self.files[index][0] == 'N' else 1

        return tile.unsqueeze(0).float(), val, self.files[index]

    def labels(self):
        labels = []
        for file in self.files:
            val = 0 if file[0] == 'N' else 1
            labels.append(val)
        return np.array(labels)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),  # I can prob just ignore this complaint
        nn.ELU(alpha=1.0, inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),  # I can prob just ignore this complaint
        nn.ELU(alpha=1.0, inplace=True))

# RESnet building

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.exspansion = 4
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.exspansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(out_channels * self.exspansion)
        self.ELU = nn.ELU(alpha=1.0, inplace=True)
        self.id = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ELU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ELU(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.id is not None:
            identity = self.id(identity)

        x += identity
        x = self.ELU(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.ELU = nn.ELU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], out_channels=514, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(514 * 4, num_classes)

    def forward(self, x):
        # x = x.half()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ELU(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def make_layer(self, block, nrb, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv3d(self.in_channels, out_channels * 4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm3d(out_channels*4))

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(nrb - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def resnet18(num_channels=1, num_classes=1):
    return ResNet(block, [2, 2, 2, 2], num_channels, num_classes)

def train(GPU=True):
    # shape = [64, 64, 64]
    test, train = getDataset()
    print(f'Test len = {len(test)} --- Train len = {len(train)}')
    testset = MRIDataset(test)
    trainset = MRIDataset(train)
    device = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')
    model = resnet18() # was (1, 64, 64, 64)

    if GPU:
        model = model.half()

        # Train model on 4 GPUs
        model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.to(device)

    # Calculate weight
    weight = getWeight()
    print(f'Weight = {weight}')
    # weight ~5. Setting pos_weight = weight should increase recall so that false negatives go down
    weight = torch.tensor([weight]).to(device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=weight)

    workers = 2 if GPU else 0

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=workers, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, eps=1e-4)

    for epoch in range(epochs):
        model.train()

        # prediction for each file in the training set and test set
        train_pre = {}
        test_pre = {}
        loss_arr = []

        # Correct prediction count for the train set and test set
        train_count = 0
        test_count = 0

        for i, (inp, target, file) in enumerate(train_loader):
            # Send inp and target to device (GPU)
            if GPU:
                inp = inp.half().to(device)
                target = target.half().to(device)

            optimizer.zero_grad()
            cur_pre = model(inp)
            loss = loss_func(cur_pre, target.unsqueeze(1).float())
            loss_arr.append(loss.item())

            # Since it is regression model until BCELogitLoss applies the sig func we need to apply sig func
            # if we want to accurately count correct predictions
            sig = nn.Sigmoid()
            sig_pre = sig(cur_pre).detach()

            for j in range(len(file)):
                train_pre[str(file[j])] = cur_pre[j].detach()

                if (target[j] == 1 and sig_pre[j] > 0.5) or (target[j] == 0 and sig_pre[j] < 0.5):
                    train_count += 1

            if not GPU:
                print("ML has started!")

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i, (inp, target, file) in enumerate(test_loader):
                # Send inp and target to device (GPU)
                if GPU:
                    inp = inp.half().to(device)
                    target = target.half().to(device)

                cur_pre = model(inp)

                sig = nn.Sigmoid()
                sig_pre = sig(cur_pre)

                for j in range(len(file)):
                    test_pre[str(file[j])] = cur_pre[j]

                    if (target[j] == 1 and sig_pre[j] > 0.5) or (target[j] == 0 and sig_pre[j] < 0.5):
                        test_count += 1

        # Calculate train and test accuracies
        train_acc = train_count / len(train_pre)
        test_acc = test_count / len(test_pre)

        # Record prediction and accuracies
        fileWriter(fold, epoch, train_pre, test_pre, train_acc, test_acc, loss_arr)

        print(f'Epoch: {epoch} finished!')

        if (epoch % 2 == 0):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(SAVE_DIR, 'Fold#' + str(fold) + 'Epoch#' + str(epoch) + 'Model.pth'))# FinalModel -> Model

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(SAVE_DIR, 'Fold#' + str(fold) + 'Model.pth'))


train(GPU=STATUS)


