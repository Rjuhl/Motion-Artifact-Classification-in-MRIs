import nibabel as nib
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import random


fold = 0
epochs = 60
batch_size = 16
g_mean = 443.8712
g_std = 743.1469
patches = 30
Status = True

BASE_DIR = "/Users/rainjuhl/Library/Application Support/JetBrains/PyCharmCE2020.1/Scratches/Dataset"
SAVE_DIR = ""
IMG_DIR = ""
MASK_DIR = "/Users/rainjuhl/PycharmProjects/MRI_BC_Project/Masks"

if Status:
    BASE_DIR = "/scratch/users/rjuhl/data2/data/Dataset"
    SAVE_DIR = "/home/users/rjuhl/MRI_BC/results1-5SKF-Kitware(older)(likelyJustBasicTiled)"
    IMG_DIR = "/scratch/users/rjuhl/imgpics"
    MASK_DIR = "/scratch/users/rjuhl/maskData/Masks"


def splitDataset():
    clean = []
    art = []
    for file in os.listdir(BASE_DIR):
        if file.startswith('.'):
            continue
        art.append(file) if str(file[0]) != 'N' else clean.append(file)
    return clean, art


def getDataset():
    clean, art = splitDataset()
    test = []
    train = []
    random.shuffle(clean)
    random.shuffle(art)
    clean_quota = {'A': 25, 'B': 33, 'C': 35, 'D': 30, 'E': 43}
    art_quota = {'A': 4, 'B': 3, 'C': 6, 'D': 2, 'E': 16}
    for file in range(len(clean)):
        if clean_quota[str(clean[file][-8])] > 0:
            test.append(clean[file])
            clean_quota[str(clean[file][-8])] = clean_quota[str(clean[file][-8])] - 1
        else:
            train.append(clean[file])
    for file in range(len(art)):
        if art_quota[str(art[file][0])] > 0:
            test.append(art[file])
            art_quota[str(art[file][0])] = art_quota[str(art[file][0])] - 1
        else:
            train.append(art[file])
    random.shuffle(test)
    random.shuffle(train)
    return test, train


def fileWriter(fold, epoch, train_pre, test_pre, train_acc, test_acc, lost_arr):
    file = open(os.path.join(SAVE_DIR, "Fold#" + str(fold) + "Epoch#" + str(epoch) + "Prediction_and_acc_scores.txt"),
                'a')
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


def tilePercentage(i_start, j_start, k_start, x_tile_size, y_tile_size, z_tile_size, masks):
    percentages = []
    for m in range(masks.shape[0]):
        mask = masks[m]
        F = 0
        T = 0
        for i in range(i_start, i_start + x_tile_size):
            for j in range(j_start, j_start + y_tile_size):
                for k in range(k_start, k_start + z_tile_size):
                    if mask[i][j][k]:
                        T += 1
                    else:
                        F += 1
        percentages.append(T / (T + F))
    return percentages


def tilesWanted(per_arr, num_wanted):
    tiles_wanted = []
    for batch in range(len(per_arr[0][1])):
        batch_tiles = []
        for per in range(len(per_arr)):
            batch_tiles.append([per_arr[per][0], per_arr[per][1][batch]])

        def sortKey(arr):
            return arr[1]

        batch_tiles.sort(key=sortKey, reverse=True)
        tiles_from_batch = []
        for patch in range(num_wanted):
            tiles_from_batch.append(batch_tiles[patch][0])

        tiles_wanted.append(tiles_from_batch)

    return tiles_wanted


def createInputTensor(tiles, inputs, index, size):
    x_size = inputs.shape[2]
    y_size = inputs.shape[3]
    z_size = inputs.shape[4]
    batch_of_tiles = [np.empty((size[0], size[1], size[2]))]

    inputs = inputs.squeeze()
    for img in range(inputs.size()[0]):
        inputs[img] = normData(inputs[img])

    print(len(inputs.size()))
    if len(inputs.size()) < 4:
        inputs = inputs.unsqueeze(0)
    inputs = inputs.unsqueeze(1)
    print(f'CIT input shape: {inputs.size()}')

    for img in range(inputs.size()[0]):
        ranges = tiles[img][index]
        print(f'CIT ranges: {ranges}')
        tile = inputs[img, :,
               ranges[0]:ranges[0] + size[0],
               ranges[1]:ranges[1] + size[1],
               ranges[2]:ranges[2] + size[2]]

        print(f'CIT Tile shape 1: {tile.size()}')
        print(f'SIZE {size[0]}, XSIZE {x_size}')
        x_pad = max(0, size[0] - x_size)
        y_pad = max(0, size[1] - y_size)
        z_pad = max(0, size[2] - z_size)

        if x_pad + y_pad + z_pad > 0:  # we need to pad
            tile = torch.nn.functional.pad(tile, (0, z_pad, 0, y_pad, 0, x_pad), 'replicate')

        tile = torch.squeeze(tile)
        print(f'CIT Tile shape 2: {tile.size()}')
        tile = tile.cpu().numpy()
        batch_of_tiles = np.append(batch_of_tiles, [tile], axis=0)

    batch_of_tiles = np.delete(batch_of_tiles, 0, axis=0)
    print(f'CIT batch of tile final size: {batch_of_tiles.shape}')
    if Status:
        return torch.from_numpy(batch_of_tiles).unsqueeze(1).float().cuda()
    else:
        return torch.from_numpy(batch_of_tiles).unsqueeze(1).float()


def getMask(file):
    arr = np.load(os.path.join(MASK_DIR, '[MASK]' + str(file) + '.npy'))
    return np.squeeze(arr)


class MRIDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = nib.load(os.path.join(BASE_DIR, self.files[index]))
        img_data = torch.squeeze(torch.from_numpy(np.array(img.get_fdata())))
        # img_data = skullstrip(img_data)
        img_data = trans(img_data)
        # img_data = normData(img_data)
        val = 0 if self.files[index][0] == 'N' else 1

        return img_data.unsqueeze(0).float(), val, self.files[index]

    def labels(self):
        labels = []
        for file in self.files:
            val = 0 if file[0] == 'N' else 1
            labels.append(val)
        return np.array(labels)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),  # I can prob just ignore this complaint
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),  # I can prob just ignore this complaint
        nn.LeakyReLU(negative_slope=0.02, inplace=True))


class Net(nn.Module):
    def __init__(self, shape):
        # 128x128x128
        super(Net, self).__init__()
        self.conv_down1 = double_conv(1, 32)
        self.conv_down2 = double_conv(32, 64)
        self.conv_down3 = double_conv(64, 128)
        self.conv_down4 = double_conv(128, 256)
        self.maxpool = nn.MaxPool3d((2, 2, 2))
        self.in_shape = shape
        self.fc1 = nn.Linear((self.in_shape[0] // 8) * (self.in_shape[1] // 8) * (self.in_shape[2] // 8) * 256, 1)

    def forward(self, x, files, firstepoch):
        if Status:
            x = x.half()

        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        x = self.conv_down4(x)
        x = x.view(-1, (self.in_shape[0] // 8) * (self.in_shape[1] // 8) * (self.in_shape[2] // 8) * 256)
        x = self.fc1(x)
        return x


class TiledClassifier(Net):
    def __init__(self, shape):
        super().__init__(shape)
        self.in_shape = shape

    def forward(self, inputs, files, firstepoch):
        # split the input image into tiles and run each tile through NN
        results = []
        x_tile_size = self.in_shape[0]
        y_tile_size = self.in_shape[1]
        z_tile_size = self.in_shape[2]
        x_size = inputs.shape[2]
        y_size = inputs.shape[3]
        z_size = inputs.shape[4]
        x_steps = math.ceil(x_size / x_tile_size)
        y_steps = math.ceil(y_size / y_tile_size)
        z_steps = math.ceil(z_size / z_tile_size)
        print([x_tile_size, y_tile_size, z_tile_size, x_size, y_size, z_size, x_steps, y_steps, z_steps])
        num = random.randint(0, 9)
        name = random.randint(0, 10000000)
        masks = np.array([getMask(files[0])])
        num_inputs = inputs.size()[0]
        print(f'Mask Arr Shape: {masks.shape}')
        for i in range(1, num_inputs):
            mask = np.array(getMask(files[i]))
            print(f'Masks Shape: {mask.shape}')
            masks = np.append(masks, [mask], axis=0)
        print(f'Final Mask Arr Shape: {masks.shape}')

        tile_percent = []
        for i in range(x_steps):
            i_start = round(i * (x_size - x_tile_size) / x_steps)
            for j in range(y_steps):
                j_start = round(j * (y_size - y_tile_size) / y_steps)
                for k in range(z_steps):
                    k_start = round(k * (z_size - z_tile_size) / z_steps)

                    # use slicing operator to make a tile
                    tile = inputs[:, :,
                           i_start:i_start + x_tile_size,
                           j_start:j_start + y_tile_size,
                           k_start: k_start + z_tile_size]

                    percent = tilePercentage(i_start, j_start, k_start, x_tile_size,
                                             y_tile_size, z_tile_size, masks)
                    tile_percent.append([[i_start, j_start, k_start], percent])

                    if num == 0 and firstepoch and Status:
                        img = nib.Nifti1Image(torch.squeeze(tile).cpu().numpy().astype(np.int32), np.eye(4))
                        nib.save(img,
                                 os.path.join(IMG_DIR, f'{str(name)}IMGTILE-I:{i_start}J:{j_start}K:{k_start}.nii.gz'))

        tw = tilesWanted(tile_percent, patches)
        for patch in range(patches):
            t = createInputTensor(tw, inputs, patch, [x_tile_size, y_tile_size, z_tile_size])
            results.append(super().forward(t, files, firstepoch))
        saveResults(results)
        print(f'RESULTS {results}')
        average = torch.mean(torch.stack(results), dim=0)
        print(f'AVERGAE {average}')

        return average

    def reset(self, shape):
        Net(self.in_shape).apply(reset_weights)


def train(GPU=True):
    shape = (64, 64, 64)
    test, train = getDataset()
    testset = MRIDataset(test)
    trainset = MRIDataset(train)
    device = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')
    model = TiledClassifier(shape=shape)  # was (1, 64, 64, 64)
    # summary(model, (1, 128, 128, 128))
    if GPU:
        model = model.half()

        # Train model on 4 GPUs
        model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.to(device)

    # Calculate weight
    clean, art = splitDataset()
    weight = len(clean) / len(art)
    # weight ~5. Setting pos_weight = weight should increase recall so that false negatives go down
    weight = torch.tensor([weight]).to(device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=weight)

    workers = 2 if GPU else 0

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=workers, shuffle=False)

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

        save = True if epoch == 0 else False

        for i, (inp, target, file) in enumerate(train_loader):
            # Send inp and target to device (GPU)
            if GPU:
                inp = inp.to(device)
                target = target.half().to(device)

            optimizer.zero_grad()
            cur_pre = model(inp, file, save)
            loss = loss_func(cur_pre, target.unsqueeze(1).float())
            loss_arr.append(loss.item())

            # Since it is regression model until BCELogitLoss applies the sig func we need to apply sig func
            # if we want to accurately count correct predictions
            sig = nn.Sigmoid()
            sig_pre = sig(cur_pre).detach()

            for i in range(len(file)):
                train_pre[str(file[i])] = cur_pre[i].detach()

                if (target[i] == 1 and sig_pre[i] > 0.5) or (target[i] == 0 and sig_pre[i] < 0.5):
                    train_count += 1

            if not GPU:
                print("ML has started!")

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for i, (inp, target, file) in enumerate(test_loader):
                # Send inp and target to device (GPU)
                if GPU:
                    inp = inp.to(device)
                    target = target.half().to(device)

                cur_pre = model(inp, file, save)

                sig = nn.Sigmoid()
                sig_pre = sig(cur_pre)

                for i in range(len(file)):
                    test_pre[str(file[i])] = cur_pre[i]

                    if (target[i] == 1 and sig_pre[i] > 0.5) or (target[i] == 0 and sig_pre[i] < 0.5):
                        test_count += 1

        # Calculate train and test accuracies
        train_acc = train_count / len(train_pre)
        test_acc = test_count / len(test_pre)

        # Record prediction and accuracies
        fileWriter(fold, epoch, train_pre, test_pre, train_acc, test_acc, loss_arr)

        print(f'Epoch: {epoch} finished!')

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(SAVE_DIR, 'Fold#' + str(fold) + 'Model.pth'))


train(GPU=Status)
