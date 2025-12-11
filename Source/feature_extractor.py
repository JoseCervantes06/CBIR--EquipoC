import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)

class MyVGG16(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        vgg16 = models.vgg16(weights='IMAGENET1K_FEATURES')
        self.model = vgg16.features
        self.model.eval()
        self.model.to(device)
        self.output_dim = 25088 

    def extract_features(self, image_tensor):
        x = IMAGENET_NORMALIZE(image_tensor)
        
        with torch.no_grad():
            feature = self.model(x)
            feature = torch.flatten(feature, start_dim=1)
        return feature.cpu().numpy()

class MyResNet50(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        modules = list(resnet50.children())[:-1]
        self.model = torch.nn.Sequential(*modules)
        self.model.eval()
        self.model.to(device)
        self.output_dim = 2048 

    def extract_features(self, image_tensor):
        x = IMAGENET_NORMALIZE(image_tensor)
        
        with torch.no_grad():
            feature = self.model(x)
            feature = torch.flatten(feature, start_dim=1)
        return feature.cpu().numpy()

class MyInceptionV3(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        inception = models.inception_v3(weights='IMAGENET1K_V1')
        inception.fc = nn.Identity()
        self.model = inception
        self.model.eval()
        self.model.to(device)
        self.output_dim = 2048 

    def extract_features(self, image_tensor):
        if image_tensor.shape[2] != 299 or image_tensor.shape[3] != 299:
            image_tensor = F.interpolate(
                image_tensor, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        x = IMAGENET_NORMALIZE(image_tensor)

        with torch.no_grad():
            feature = self.model(x)
            feature = torch.flatten(feature, start_dim=1)
        return feature.cpu().numpy()
    
class MyColorHistogram(nn.Module):
    def __init__(self, bins=64, device=None):
        super().__init__()
        self.bins = bins
        self.device = device
        self.output_dim = bins * 3 

    def extract_features(self, image_tensor):
        features_list = []
        
        image_tensor = torch.clamp(image_tensor, 0, 1)

        for img in image_tensor:
            img_features = []
            for channel_idx in range(3):
                channel = img[channel_idx]
                hist = torch.histc(channel, bins=self.bins, min=0.0, max=1.0)
                hist = hist / (hist.sum() + 1e-7)
                img_features.append(hist)
            
            features_list.append(torch.cat(img_features))

        features = torch.stack(features_list)
        return features.cpu().numpy()
    
class MyLBP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.n_points = 24
        self.radius = 3
        self.shape = self.n_points + 2 

    def extract_features(self, images):

        images_np = images.cpu().numpy().clip(0, 1)

        features_list = []

        for img in images_np:
            # Transponer de (C, H, W) -> (H, W, C)
            img_uint8 = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            # Convertir a Gris 
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

            # Calcular LBP
            lbp = local_binary_pattern(gray, self.n_points, self.radius, method="uniform")

            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

            if len(hist) < self.shape:
                hist = np.pad(hist, (0, self.shape - len(hist)), 'constant')
            
            hist = hist.astype("float32")
            hist /= (hist.sum() + 1e-7)

            features_list.append(hist)

        return torch.tensor(np.array(features_list), dtype=torch.float32).to(self.device)