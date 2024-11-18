"""
Pose Loss for rotation and translation prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import MODELS

@MODELS.register_module()
class PoseLoss(nn.Module):
    def __init__(self, 
                 rotation_weight=1.0,      # rotation loss의 전체 가중치
                 translation_weight=1.0,    # translation loss의 전체 가중치
                 static_weight=2.0,        # 정적 부분에 대한 가중치
                 epsilon=1e-6,             # 수치적 안정성을 위한 작은 상수
                 static_classes=None):      # config에서 받아옴
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.static_weight = static_weight
        self.epsilon = epsilon
        self.static_classes = static_classes

    def forward(self, pred_rot, pred_trans, target_rot, target_trans, segment_labels):
        # segment_labels로부터 static_mask 생성
        static_mask = torch.zeros_like(segment_labels, dtype=torch.bool)
        for class_id in self.static_classes:
            static_mask |= (segment_labels == class_id)

        # Translation Loss 계산
        trans_loss = torch.sqrt(F.mse_loss(pred_trans, target_trans, reduction='none') + self.epsilon)
        
        # Rotation Loss 계산
        rot_loss = torch.sqrt(F.mse_loss(pred_rot, target_rot, reduction='none') + self.epsilon)
        
        # 정적/동적 가중치 적용
        weight = torch.where(static_mask, self.static_weight, 1.0)
        
        # 가중치가 적용된 최종 loss 계산
        total_trans_loss = (trans_loss * weight).mean() * self.translation_weight
        total_rot_loss = (rot_loss * weight).mean() * self.rotation_weight
        
        # 논문의 수식대로 최종 loss 계산
        Sx = -torch.log(total_trans_loss)
        Sq = -torch.log(total_rot_loss)
        total_loss = torch.exp(-Sx) * total_trans_loss + Sx + \
                    torch.exp(-Sq) * total_rot_loss + Sq
                
        return total_loss