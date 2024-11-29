import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import random

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


import random
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def random_activation():
    # Random választás a ReLU és LeakyReLU között
    activation_choice = random.choice(['ReLU', 'LeakyReLU'])

    if activation_choice == 'ReLU':
        print("Selected activation: ReLU")
        return nn.ReLU()
    else:
        # Generálunk egy véletlen értéket a megadott tartományon belül
        negative_slope = random.uniform(0.005, 0.03)
        print(f"Selected activation: LeakyReLU with negative_slope={negative_slope}")
        return nn.LeakyReLU(negative_slope=negative_slope)


class FocusNetPretrained(nn.Module):
    def __init__(self):
        super(FocusNetPretrained, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Rétegek fagyasztása
        for name, param in self.model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Dropout érték véletlenszerű kiválasztása
        dropout_value = random.choice([round(0.4 + i * 0.1, 1) for i in range(-2, 3)])
        print(f"Dropout value: {dropout_value}")

        # Model komponensek inicializálása
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            random_activation(),  # Véletlenszerű aktivációs réteg
            nn.Dropout(dropout_value),  # Dinamikus Dropout
            nn.Linear(512, 1),
        )

    #  self.model.fc = nn.Sequential(       #RGB
       #     nn.Linear(self.model.fc.in_features, 512),
        #    nn.ReLU(),
         #   nn.Dropout(0.4),
          #  nn.Linear(512, 1)  # Egy kimenet a regresszióhoz"""

    def forward(self, x):
        return self.model(x)
