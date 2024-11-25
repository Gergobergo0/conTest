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

        # RGB



        # Rétegek fagyasztása
        # Frozen weights (only train last layers)
        for name, param in self.model.named_parameters():
            param.requires_grad = True

        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

        # Unfreeze all layers for fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True



            
            
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.35),  # Magasabb dropout a túltanulás ellen
            nn.Linear(512, 1),
            )





      #  self.model.fc = nn.Sequential(       #RGB
       #     nn.Linear(self.model.fc.in_features, 512),
        #    nn.ReLU(),
         #   nn.Dropout(0.4),
          #  nn.Linear(512, 1)  # Egy kimenet a regresszióhoz"""

    def forward(self, x):
        return self.model(x)
