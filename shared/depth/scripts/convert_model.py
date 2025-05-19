import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
import cv2
from shared.depth.scripts.transform import Resize, NormalizeImage, PrepareForNet

torch.manual_seed(0)


class MobileNetSkipAdd(nn.Module):
    def __init__(self):
        super(MobileNetSkipAdd, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 56, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(56, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=56, bias=False),
            nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(56, 88, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False),
            nn.BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(88, 120, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(120, 120, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=120, bias=False),
            nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(120, 144, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(144, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(256, 408, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=408, bias=False),
            nn.BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(408, 376, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(376, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(376, 376, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=376, bias=False),
            nn.BatchNorm2d(376, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(376, 272, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(272, 272, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=272, bias=False),
            nn.BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(272, 288, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False),
            nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(288, 296, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(296, 296, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=296, bias=False),
            nn.BatchNorm2d(296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(296, 328, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(328, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(328, 328, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=328, bias=False),
            nn.BatchNorm2d(328, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(328, 480, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False),
            nn.BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
            nn.Conv2d(480, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU6(inplace=True),
        )
        self.decode_conv1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 200, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(200, 200, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=200, bias=False),
                nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(200, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv3 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 120, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv4 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False),
                nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(120, 56, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv5 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(56, 56, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=56, bias=False),
                nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(56, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
        )
        self.decode_conv6 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i == 4:
                if x.shape != x1.shape:
                    min_h = min(x.shape[2], x1.shape[2])
                    min_w = min(x.shape[3], x1.shape[3])
                    x = x[:, :, :min_h, :min_w]
                    x1 = x1[:, :, :min_h, :min_w]
                x = x + x1
                print(f"After addition: x shape: {x.shape}")
            elif i==3:
                x = x + x2
                print(f"x shape: {x.shape}, x2 shape: {x2.shape}")
            elif i==2:
                if x.shape != x3.shape:
                    min_h = min(x.shape[2], x3.shape[2])
                    min_w = min(x.shape[3], x3.shape[3])
                    x = x[:, :, :min_h, :min_w]
                    x3 = x3[:, :, :min_h, :min_w]
                x = x + x3
                print(f"x shape: {x.shape}, x3 shape: {x3.shape}")
        x = self.decode_conv6(x)
        return x
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)

# Tracing the module
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
m = MobileNetSkipAdd().to(DEVICE)

m.load_state_dict(torch.load("/scratch/kris/RoboRT_Experiment/shared/depth/checkpoints/mobilenet_skip_add.pth", map_location=torch.device('cpu')))
m.eval()

# Example usage of image2tensor
input_file = "/scratch/kris/RoboRT_Experiment/shared/depth/data/visual_1.png"
img = cv2.imread(input_file)

if img is None:
    raise FileNotFoundError(f"Image not found at {input_file}")

depth_input, (h, w) = m.image2tensor(img)

depth_input = depth_input.to(DEVICE)

print(depth_input)

with torch.no_grad():
    print(f"input shape: {depth_input.shape}")
    output = m.forward(depth_input)
    print(f"output shape: {output.shape}")
    depth_image = output  # Assign the output to depth_image
    if len(depth_image.shape) == 3:
        depth_image = depth_image.unsqueeze(0)
    depth_image = torch.nn.functional.interpolate(
        depth_image, size=(518, 910), mode='bilinear', align_corners=False
    )
    print(depth_image)
