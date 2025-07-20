#!/usr/bin/env python3
"""
Quick TORUS Test - Демонстрация работоспособности
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn

class MockUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 3
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        
    def forward(self, sample, timestep, return_dict=True):
        x = self.conv(sample)
        if return_dict:
            return type('Output', (), {'sample': x})()
        return x

def quick_test():
    print("=== БЫСТРЫЙ ТЕСТ TORUS ===")
    
    # Импортируем TORUS
    from toroidal_diffusion_wrapper import ToroidalDiffusionModel
    
    # Создаем модель
    base_model = MockUNet()
    scheduler = type('Scheduler', (), {'timesteps': torch.tensor([1])})()
    
    model = ToroidalDiffusionModel(
        base_model=base_model,
        scheduler=scheduler,
        enable_singularity=True,
        enable_coherence_monitoring=True
    )
    
    print(f"✅ Модель создана: {sum(p.numel() for p in model.parameters()):,} параметров")
    
    # Тестируем forward pass
    test_input = torch.randn(1, 3, 32, 32)
    output = model(test_input, torch.tensor([500]))
    
    print(f"✅ Forward pass: {output['sample'].shape}")
    print("🎉 TORUS РАБОТАЕТ!")
    
    return True

if __name__ == "__main__":
    quick_test() 