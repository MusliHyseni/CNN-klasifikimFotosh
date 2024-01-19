"""
Ky është një rrjet neural për klasifikimin e objekteve
"""

# Marrim libraritë e nevojitura
import torch

import torch.optim as optim

from torch import nn
import torch.nn.functional as F

import torch.utils
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Përkufizojmë hiperparametrat
# Numri i sub-proceseve në marrjen e të dhënave
num_workers = 0

# Numri i shembujve për një bashkësi ("grumbull")
batch_size = 20

# Përqindja e bashkësisë së të dhënave, e cila përdoret për validim
valid_size = 0.2

# Transformimi i input-it (fotove)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Të dhënat për trajnim
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)

# Të dhënat për testim
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# Numri i shembujve për trajnim
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

# Ndajmë dataset-in (0.8 ushtrim, 0.2 validim)
split = int(np.floor(valid_size * num_train))
# Ndarja
train_idx, valid_idx = indices[split:], indices[:split]

# Shembull-marrësit e përdorur për marrjen e "grumbujve"
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(valid_idx)

# Marrësit e të dhënave
train_loader = torch.utils.data.DataLoader(train_data, batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
valid_loader = torch.utils.data.DataLoader(test_data, batch_size, sampler=test_sampler, num_workers=num_workers, drop_last=True)

# Llojet(klasat) e fotove
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Funksioni për vizualizimin e fotove
def show_image(image):
    image = image / 2 + 0.5
    plt.imshow(np.transpose(image, (1, 2, 0)))

# I bëjmë të dhënat të iterueshme
images = []
iterable_data = iter(train_loader)
for i in range(20):
    image, label = train_loader.dataset[i]
    # Konvertojmë foton në numpy, për ta paraqitur
    image = image.numpy()
    # Vendosim foton në listën e fotove
    images.append(image)


# Përkufizojmë kornizën
fig = plt.figure(figsize=(25, 4))

# Fusim fotot në kornizë
for idx in np.arange(20):
    visual = fig.add_subplot(2, 10, idx+1)
    show_image(images[idx])


# Modeli konvolucional

# Në PyTorch, modelet trashëgojnë nn.Module
class CIFARClassifier(nn.Module):
    def __init__(self):
        super(CIFARClassifier, self).__init__()
        self.conv_layer = nn.Sequential(
            # Blloku parë konvolucional
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blloku i dytë konvolucional
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),

            # Blloku i tretë konvolucional
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Shresa e lidhur
        self.fully_connected_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    # Forward-propagation
    def forward(self, x):

        # Shtresat konvolucionale
        x = self.conv_layer(x)

        # Rrafshimi
        x = x.view(x.size(0), -1)

        # Shtresa e lidhur
        x = self.fully_connected_layer(x)

        return x


# Krijojmë modelin si instancë të rrjetit neural
model = CIFARClassifier()
print(model)

# Kalojmë modelin në GPU, nëse ka mundësi

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
# Numri i iterimeve të trajnimit
n_epochs = 20

# Lista ku ruhet humbja (për vizualizim)
train_loss_list = []

# Ndryshimet në humbje gjatë validimit
valid_loss_min = np.Inf

for epoch in range(1, n_epochs+1):
    # Humbja inicializohet si zero
    train_loss, valid_loss = 0, 0

    # ------------- Ushtrojmë modelin --------------

    model.train()

    # data: të dhënat për trajnim, target: të dhënat e sakta (vlerat të cilat dëshirojmë t'i parashikojmë)
    for data, target in train_loader:
        # Kalojmë tensorët në GPU, nëse ka mundësi (nëse gjendet në dispozicion)


        # Fshij gradientet e ndryshoreve të optimizuara nga iterimi i fundit i trajnimit
        optimizer.zero_grad()

        # Shtyj inputet përpara (në rrjet neural) - forward-propagation
        output = model(data)

        # Llogarisim humbjen (sa jemi të pasaktë)
        loss = loss_fn(output, target)

        # Shtyjmë prapa: llogarisim gradientin e humbjes, me respekt ndaj parametrave të modelit
        loss.backward()

        # Hapi optimizues
        optimizer.step()

        # Ripërtrijmë humbjen gjatë trajnimit
        train_loss += loss.item()*data.size(0)


    # ------------- Ushtrojmë modelin --------------

    model.eval()

    for data, target in valid_loader.dataset:
        target = torch.tensor([target])
        # Kalojmë tensorët në GPU, nëse ka mundësi (nëse gjendet në dispozicion)


        # Shtyj inputet përpara (në rrjet neural) - forward-propagation
        data = data.unsqueeze(0)
        output = model(data)

        # Llogarisim humbjen (sa jemi të pasaktë)
        loss = loss_fn(output, torch.tensor([target]))

        # Ripërtrijmë humbjen gjatë trajnimit
        valid_loss += loss.item()*data.size(0)

    # Llogarisim humbjen mesatare
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    # Vendosim himbjen në array
    train_loss_list.append(train_loss)

    # Paraqesim statistikat e trajnimit dhe validimit
    print('Iterimi: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # Ruaj modelin nëse humbja e validimit është zvogëluar
    if valid_loss <= valid_loss_min:
        print('Humbja e validimit është zvogëluar ({:.6f} --> {:.6f}).  Modeli po ruhet ...'.format(
        valid_loss_min,
        valid_loss))

        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

plt.plot(range(1, n_epochs+1), train_loss_list, label='Training Loss')
plt.xlabel("Epoch (iterimi)")
plt.ylabel("Humbja")
plt.title("Performanca e modelit")
plt.show()
