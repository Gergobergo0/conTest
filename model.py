import torch
import torch.nn as nn
from torchvision.models import resnet18

class FocusNet(nn.Module):
    def __init__(self):
        super(FocusNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class FocusNetPretrained(nn.Module):
    def __init__(self):
        super(FocusNetPretrained, self).__init__()
        self.model = resnet18(pretrained=True)
        for name, param in self.model.named_parameters():
            if "layer4" not in name:  # A 'layer4' és a teljesen összekötött rétegek tanulhatnak
                param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Output layer módosítása

    def forward(self, x):
        return self.model(x)
