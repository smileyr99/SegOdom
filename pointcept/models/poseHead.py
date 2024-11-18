"""
Pose Head for rotation and translation prediction.
"""
import torch
import torch.nn as nn
from .builder import MODELS

@MODELS.register_module()
class PoseHead(nn.Module):
    def __init__(
        self,
        in_features,           # 입력 특징 차원
        hidden_features=100,   # 중간 레이어 차원
        activation='relu',     # 활성화 함수 
    ):
        super().__init__()
        
        # 활성화 함수 설정
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        
        # Rotation MLP (quaternion prediction)
        self.rotation_mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            self.activation,
            nn.Linear(in_features=hidden_features, out_features=4)  # quaternion (x,y,z,w)
        )
        
        # Translation MLP
        self.translation_mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            self.activation,
            nn.Linear(in_features=hidden_features, out_features=3)  # translation (x,y,z)
        )

        print("PoseHead Parameters:")
        print(f"Number of trainable parameters in rotation MLP: {sum(p.numel() for p in self.rotation_mlp.parameters()):,}")
        print(f"Number of trainable parameters in translation MLP: {sum(p.numel() for p in self.translation_mlp.parameters()):,}")

    def forward(self, x):
        # Rotation prediction and normalization
        rot_pred = self.rotation_mlp(x)
        rot_pred = rot_pred / torch.norm(rot_pred, dim=1, keepdim=True)  # normalize quaternion
        
        # Translation prediction
        trans_pred = self.translation_mlp(x)
        
        return rot_pred, trans_pred