import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, z_dim, label_num=10, device='cpu'):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.label_num = label_num
        net = []
        net.append(nn.Linear(self.z_dim + self.label_num, 1024))
        net.append(nn.ReLU())
        net.append(nn.Linear(1024, 7 * 7 * 32))
        net.append(nn.ReLU())
        net.append(nn.Linear(7 * 7 * 32, 28 * 28 * 1))
        net.append(nn.Tanh())
        self.fc = nn.Sequential(*net).to(device)

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, label_num=10, device='cpu'):
        super(Discriminator, self).__init__()
        self.label_num = label_num
        net = []
        net.append(nn.Linear(28 * 28 * 1 + self.label_num, 1024))
        net.append(nn.LeakyReLU(0.2))
        net.append(nn.Linear(1024, 256))
        net.append(nn.LeakyReLU(0.2))
        net.append(nn.Linear(256, 1))
        net.append(nn.Sigmoid())
        self.fc = nn.Sequential(*net).to(device)

    def forward(self, x):
        return self.fc(x)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad(opt_d, opt_g):
    opt_d.zero_grad()
    opt_g.zero_grad()

epoch_num = 100
batch_size = 128
z_dim = 100
label_num = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],   # 1 for greyscale channels
                                     std=[0.5])])

train_data = datasets.MNIST('./Data/', train=True, download=False, transform=transform)
test_data = datasets.MNIST('./Data/', train=False, download=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

criterion = nn.BCELoss()

D = Discriminator(label_num=label_num, device=device)
G = Generator(z_dim=z_dim, label_num=label_num, device=device)

optimizer_d = torch.optim.Adam(D.parameters(), lr=0.0002)
optimizer_g = torch.optim.Adam(G.parameters(), lr=0.0002)


train_step = len(train_loader)
for epoch in range(epoch_num):
    for i, (images, label) in enumerate(train_loader):
        images = images.reshape(batch_size, -1).to(device)
        onehot_label = torch.zeros(batch_size, label_num).scatter_(1, label.view(-1, 1), 1).to(device)

        x = torch.cat((images, onehot_label), 1).to(device)
        
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        outputs = D(x)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, z_dim).to(device)
        z = torch.cat((z, onehot_label), 1).to(device)
        fake_images = G(z)
        outputs = D(torch.cat((fake_images, onehot_label), 1).to(device))
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake

        reset_grad(optimizer_d, optimizer_g)
        d_loss.backward()
        optimizer_d.step()

        
        z = torch.randn(batch_size, z_dim).to(device)
        z = torch.cat((z, onehot_label), 1).to(device)
        fake_images = G(z)
        outputs = D(torch.cat((fake_images, onehot_label), 1).to(device))

        g_loss = criterion(outputs, real_labels)

        reset_grad(optimizer_d, optimizer_g)
        g_loss.backward()
        optimizer_g.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                  % (epoch, epoch_num, i + 1, train_step, d_loss.item(), g_loss.item(),
                     real_score.mean().item(), fake_score.mean().item()))
        
    if (epoch+1) == 1:
        images = images.reshape(batch_size, 1, 28, 28)
        save_image(denorm(images), './cgan_result/real_images.png')
    
    fake_images = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(denorm(fake_images), './cgan_result/fake_images_%d.png' % (epoch+1))
