import torch
import torch.nn.functional as F  # one_hot 함수 사용을 위해 추
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)  # backbone 모델 생성
        self.criteria = build_criteria(criteria)  # 손실 함수(criterion) 생성

    def forward(self, input_dict):
        if "condition" in input_dict.keys():  
            # PPT 논문에서 언급된 조건 적용 (https://arxiv.org/abs/2308.09718)
            # 현재는 한 배치에 하나의 조건만 지원
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)  # backbone을 통해 segmentation logits 계산
        # 학습 단계일 경우
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])  # 손실 함수 계산
            return dict(loss=loss)
        # 평가 단계일 경우
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])  # 손실 함수 계산
            return dict(loss=loss, seg_logits=seg_logits)
        # 테스트 단계일 경우
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        pose_head=None,
    ):
        super().__init__()
        # segmentation head 설정
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)  
        self.criteria = build_criteria(criteria)  
        self.num_classes = num_classes
        
        # pose estimation을 위한 head 추가
        self.pose_head = build_model(pose_head)

    def process_single_frame(self, coord, strength):
        """단일 프레임 처리를 위한 helper 함수"""
        point = Point({'coord': coord, 'strength': strength})
        point = self.backbone(point)
        feat = point.feat if isinstance(point, Point) else point
        seg_logits = self.seg_head(feat)
        return feat, seg_logits

    def process_frame_pair(self, input_dict):
        """t, t-1 프레임 쌍 처리를 위한 helper 함수"""
        # 1. t-1 시점 처리
        feat_t1, seg_logits_t1 = self.process_single_frame(
            input_dict['prev_coord'], 
            input_dict['prev_strength']
        )

        # 2. t 시점 처리
        feat_t, seg_logits_t = self.process_single_frame(
            input_dict['coord'], 
            input_dict['strength']
        )

        # 3. feature와 label 결합
        sem_one_hot_t1 = F.one_hot(torch.argmax(seg_logits_t1, dim=1), 
                                  num_classes=self.num_classes).float()
        sem_one_hot_t = F.one_hot(torch.argmax(seg_logits_t, dim=1), 
                                 num_classes=self.num_classes).float()

        combined_t1 = torch.cat([feat_t1, sem_one_hot_t1], dim=1)
        combined_t = torch.cat([feat_t, sem_one_hot_t], dim=1)

        return feat_t1, feat_t, seg_logits_t1, seg_logits_t, combined_t1, combined_t

    def forward(self, input_dict):
        if self.training:
            if "segment" in input_dict.keys():
                # Semantic Segmentation 학습
                _, _, seg_logits_t1, seg_logits_t, _, _ = self.process_frame_pair(input_dict)
                loss_t1 = self.criteria(seg_logits_t1, input_dict["prev_segment"])
                loss_t = self.criteria(seg_logits_t, input_dict["segment"])
                return dict(loss_t1=loss_t1, loss_t=loss_t)
            
            else:
                # Pose Estimation 학습
                _, _, _, _, combined_t1, combined_t = self.process_frame_pair(input_dict)
                final_feature = torch.cat([combined_t, combined_t1], dim=1)
                rot_pred, trans_pred = self.pose_head(final_feature)
                return dict(rot_pred=rot_pred, trans_pred=trans_pred)
        else:
            # Inference/Test 모드
            _, _, seg_logits_t1, seg_logits_t, combined_t1, combined_t = self.process_frame_pair(input_dict)
            final_feature = torch.cat([combined_t, combined_t1], dim=1)
            rot_pred, trans_pred = self.pose_head(final_feature)

            if "segment" in input_dict.keys():
                # 평가 단계
                loss_t1 = self.criteria(seg_logits_t1, input_dict["prev_segment"])
                loss_t = self.criteria(seg_logits_t, input_dict["segment"])
                return dict(
                    loss_t1=loss_t1,
                    loss_t=loss_t,
                    seg_logits_t=seg_logits_t,
                    seg_logits_t1=seg_logits_t1,
                    rot_pred=rot_pred,
                    trans_pred=trans_pred
                )
            else:
                # 테스트 단계
                return dict(
                    seg_logits_t=seg_logits_t,
                    seg_logits_t1=seg_logits_t1,
                    rot_pred=rot_pred,
                    trans_pred=trans_pred
                )


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)  # backbone 모델 생성
        self.criteria = build_criteria(criteria)  # 손실 함수 생성
        self.num_classes = num_classes  # 클래스 수 설정
        self.backbone_embed_dim = backbone_embed_dim  # backbone의 임베딩 차원 설정
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),  # 임베딩 차원에서 256차원으로 변경
            nn.BatchNorm1d(256),  # 배치 정규화 적용
            nn.ReLU(inplace=True),  # ReLU 활성화 함수
            nn.Dropout(p=0.5),  # 드롭아웃 적용
            nn.Linear(256, 128),  # 128차원으로 축소
            nn.BatchNorm1d(128),  # 배치 정규화
            nn.ReLU(inplace=True),  # ReLU 활성화
            nn.Dropout(p=0.5),  # 드롭아웃 적용
            nn.Linear(128, num_classes),  # 최종 클래스 개수로 매핑
        )

    def forward(self, input_dict):
        point = Point(input_dict)  # 입력 데이터를 Point 구조로 변환
        point = self.backbone(point)  # backbone을 통해 Point 처리
        # v1.5.0 이후 backbone이 Point 구조를 반환
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )  # feature를 segment 단위로 평균하여 집계
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)  # 분류 logits 계산
        # 학습 단계일 경우
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])  # 손실 함수 계산
            return dict(loss=loss)
        # 평가 단계일 경우
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])  # 손실 함수 계산
            return dict(loss=loss, cls_logits=cls_logits)
        # 테스트 단계일 경우
        else:
            return dict(cls_logits=cls_logits)
