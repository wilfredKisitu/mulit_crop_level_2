import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.Residual_Block import ResidualBlock


class Resnet(nn.Module):

    def __init__(self, num_plants, num_diseases):
        """Implementation of the custom Resnet for model training"""
        super(Resnet, self).__init__()

        # Block A 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1) # B
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2) # C
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2) # D
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2) # E

        self.plant_head = nn.Linear(512, num_plants)
        self.disease_head = nn.Linear(512, num_diseases)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Builds channels while ensuring shape consistency"""
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        plant_out = self.plant_head(out)
        disease_out = self.disease_head(out)
        return plant_out, disease_out
    
if __name__ == "__main__":
    import os 

    model = Resnet(num_plants=5, num_diseases=38)
    model.eval()

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        p_t, p_d = model(x)

    os.system('clear')
    print(f'Input shape: {x.shape}')
    print(f'Plant type shape: {p_t.shape}')
    print(f'Disease type shape: {p_d.shape}')
    
    

