
xyz=1
import torch
import torch.nn as nn
#    import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

import timm
import os
import shutil
import random as rn

from datetime import datetime
from PIL import Image
import pretrainedmodels

device = 'cuda'
path = './input/val'
path2 = './input/unmask_sample/'
path3 = './input/Mask_sample/'
path4 = './input/test1'


class MaskunmaskDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            if 'unmask' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]

def check_accuracy(model, loader, dtype):
    model.eval()
    num_correct, num_samples = 0, 0
    for x, y in loader:
        x_var = x.type(dtype)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += x.size(0)

    acc = float(num_correct) / num_samples
    return acc


def countunmaskorMask(name):

    path = './input/train'
    counter = 0
    for x in os.listdir(path):
        if name in x:
            counter += 1
        else:
            continue
    return counter



def changenamestonormal(name):

    path = './input/train'
    count = 0
    listt = os.listdir(path)
    os.chdir(path)
    rn.shuffle(listt)
    for x in listt:
        if name in x:

            try:
                os.rename(x, str(name + 's') + str(count) + '.jpg')
            except FileExistsError:
                print(name + ' file already exists')

            # os.rename(str(name+'s')+str(count)+'.jpg',str(name)+str(count)+'.jpg')
            count += 1
        else:
            continue
    count = 0
    listt = os.getcwd()
    listt=os.listdir((listt))
    for x in listt:
        if name in x:
            try:
                os.rename(x, str(name) + str(count) + '.jpg')
            except FileExistsError:
                print(name + ' file already exists')

            count += 1
        else:
            continue
    os.chdir('../')
    os.chdir('../')
    os.chdir('./')
train_dir = './input/train'
test_dir = './input/test1'
val_dir = './input/val'
for x in os.listdir(path):
    if '.jpg' in x:
        shutil.move('./input/val/{}'.format(x),
                    './input/train')
for x in os.listdir(path2):
    if '.jpg' in x:
        shutil.move('./input/unmask_sample/{}'.format(x),
                    './input/train')
for x in os.listdir(path3):
    if '.jpg' in x:
        shutil.move('./input/Mask_sample/{}'.format(x),
                    './input/train')
for x in os.listdir(path4):
    if '.jpg' in x:
        shutil.move('./input/test1/{}'.format(x),
                    './input/train')

changenamestonormal('Mask')
changenamestonormal('unmask')

Maskmax = countunmaskorMask('Mask')
unmaskmax = countunmaskorMask('unmask')

os.chdir('.')

leastClassValue = min(len([x for x in os.listdir('./input/train') if 'Mask' in x]),
                      len([x for x in os.listdir('./input/train') if 'unmask' in x]))
train_valu = round(leastClassValue * .001)


val_value = leastClassValue - train_valu

Mask_val_we_want_to_train = train_valu
Mask_sample = rn.sample(range(1, Maskmax), Maskmax - Mask_val_we_want_to_train)
unmask_sample = rn.sample(range(1, unmaskmax), unmaskmax - Mask_val_we_want_to_train)
not_Masksample = [x for x in range(Maskmax) if x not in Mask_sample]
not_unmasksample = [x for x in range(unmaskmax) if x not in unmask_sample]
sam = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
sam = sam.replace(':', '-')
# f = open('saved\{}.txt'.format(sam), 'w')
# f.write('Mask' + str(not_Masksample))
# f.write('\n\n unmask' + str(not_unmasksample))
# f.close()
for Mask_val in Mask_sample:
    shutil.move('./input/train/Mask{}.jpg'.format(Mask_val),
                './input/Mask_sample/')
for unmask_val in unmask_sample:
    shutil.move('./input/train/unmask{}.jpg'.format(unmask_val),
                './input/unmask_sample/')
for x in range(val_value):
    sample = unmask_sample[x]
    shutil.move('./input/unmask_sample/unmask{}.jpg'.format(sample),
                './input/val/')

for x in range(val_value):
    sample = Mask_sample[x]
    shutil.move('./input/Mask_sample/Mask{}.jpg'.format(sample),
                './input/val/')


train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
val_files = os.listdir(val_dir)

# data_transform = transforms.Compose([
#     transforms.Resize(331),
#     transforms.ColorJitter(),
#     # transforms.RandomCrop(300),
#     transforms.RandomCrop(300, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(331),
#
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#
# ])

data_transform = transforms.Compose([
    transforms.Resize(331),

    # transforms.RandomCrop(300),



    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
test_transform = transforms.Compose([
    transforms.Resize((331, 331)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

Mask_files = [tf for tf in train_files if 'Mask' in tf]
unmask_files = [tf for tf in train_files if 'unmask' in tf]

Mask_files_val = [tf for tf in val_files if 'Mask' in tf]
unmask_files_val = [tf for tf in val_files if 'unmask' in tf]

Masks = MaskunmaskDataset(Mask_files, train_dir, transform=data_transform)
unmasks = MaskunmaskDataset(unmask_files, train_dir, transform=data_transform)

Masks_val = MaskunmaskDataset(Mask_files_val, val_dir, transform=test_transform)
unmasks_val = MaskunmaskDataset(unmask_files_val, val_dir, transform=test_transform)
#
Maskunmasks = ConcatDataset([Masks, unmasks])
Maskunmasks_val = ConcatDataset([Masks_val, unmasks_val])

dataloader = DataLoader(Maskunmasks, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(Maskunmasks_val, batch_size=1, shuffle=True, num_workers=0)

device = 'cuda'

# model_name = 'nasnetalarge'
# model_name = 'pnasnet5large'
# # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# # model = pretrainedmodels.__dict__[model_name](num_classes=1000)
# model = pretrainedmodels.nasnetalarge
#        print(pretrainedmodels.pretrained_settings[model_name])
# model = EfficientNet.from_pretrained('efficientnet-b0')

# model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=1000)
model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)

model.classifier = nn.Sequential(
    nn.Linear(2560, 500),
    nn.Linear(500, 2)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

dtype = torch.cuda.FloatTensor
epochs = 10000
itr = 1
# p_itr=800 for 3000
p_itr = 1250
# p_itr = 200 for 1400
model.train()
total_loss = 0
loss_list = []
acc_list = []
val_list = []

val_accuracy = 0.7
save_val = 0
winner_counter = 0
current_winner = 0
flag = 0

for epoch in range(epochs):

    for samples, labels in dataloader:
        ## added zeros to all

        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(samples)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step()

        if itr % p_itr == 0:
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch + 1, epochs, itr,
                                                                                              total_loss / p_itr, acc))
            # loss_list.append(total_loss/p_itr)
            val_acc = check_accuracy(model, val_loader, dtype)
            if itr > 1400:
                loss_list.append(total_loss / p_itr)
                val_list.append(val_acc)
                acc_list.append(acc)
                avgval = round(sum(val_list) / (len(val_list) + 0.000001), 2)
                if len(val_list) == 1:
                    current_winner = avgval
                if avgval > current_winner:
                    current_winner = avgval

                    print('we have a current winner! {:.3}'.format(current_winner))
                    winner_counter = 0
                else:
                    winner_counter += 1
                    print('we have a loser! {}++{}'.format(winner_counter, avgval))

            print("val accuracy")
            plt.plot(val_list)

            # print("training loss")
            plt.plot(loss_list)
            plt.legend(['val_acc', 'training_loss'])

            plt.show()
            total_loss = 0

            print('Val accuracy: ', val_acc)

            if itr > (10000 / 2):
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                current_time = current_time.replace(':', '-')

                filename_pth = 'saved\{}PNASNET_pinay_test{}+{}+{:.3f}+{}+avgval{:.4f}.pth'.format(xyz, save_val, itr,
                                                                                                   val_acc,
                                                                                                   current_time, avgval)
                # filename_pth = 'saved\{}PNASNET_pinay_test{}+{}+{:.3f}+{}+avgval{:.4f}.pth'.format(xyz,save_val,itr,val_acc,current_time,avgval)
                torch.save(model.state_dict(), filename_pth)

            model.train()

        itr += 1