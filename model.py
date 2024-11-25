import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


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
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Első konvolúciós réteg módosítása a 3 bemeneti csatornára (_amp, _phase, _mask_)
        self.model.conv1 = nn.Conv2d(
            in_channels=3,  # Három csatorna: _amp, _phase, _mask_
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False
        )

        # Rétegek fagyasztása (opcionális)
        for name, param in self.model.named_parameters():
            if "layer4" not in name:  # Csak a magasabb rétegek tanulhatnak
                param.requires_grad = False

        # Fine-tuning: Minden réteg tanítható
        for param in self.model.parameters():
            param.requires_grad = True

        # Kimeneti réteg konfigurálása (regresszió)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout a túltanulás ellen
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Egy kimenet a regresszióhoz
        )




      #  self.model.fc = nn.Sequential(       #RGB
       #     nn.Linear(self.model.fc.in_features, 512),
        #    nn.ReLU(),
         #   nn.Dropout(0.4),
          #  nn.Linear(512, 1)  # Egy kimenet a regresszióhoz"""

    def forward(self, x):
        return self.model(x)
