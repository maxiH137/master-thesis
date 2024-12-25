import torch
import numpy as np

class DPrivacy():
    def __init__(self, multiplier = None, clip = None):
        self.multiplier = multiplier
        self.clip = clip
        
    def addNoise(self, gradient):
        return gradient + torch.normal(0, self.multiplier, gradient.shape)
    
    def clipGradient(self, gradient):
        return torch.clamp(gradient, -self.clip, self.clip)