import torch
from pointcept.datasets.builder import build_dataset
from pointcept.models.builder import build_model
from pointcept.models.utils.structure import Point

# 데이터셋 설정
dataset_cfg = {
    "type": "SemanticKITTIDataset",
    "split": "train",
    "data_root": "data/semantic_kitti",
    "transform": [
        {"type": "ToTensor"},
    ],
    "test_mode": False
}

try:
    dataset = build_dataset(dataset_cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print("\n데이터셋과 DataLoader가 성공적으로 생성되었습니다.")
except Exception as e:
    print(f"데이터셋 또는 DataLoader 생성 중 에러 발생: {e}")
    exit()

# 모델 설정
model_cfg = {
    "type": "DefaultSegmentorV2",
    "num_classes": 19,
    "backbone_out_channels": 64,
    "backbone": {"type": "PT-v3m1"},  # PointTransformer가 registry에 등록되었는지 확인 필요
    "criteria": [{"type": "CrossEntropyLoss"}],
    "pose_head": {
        "type": "PoseHead",
        "rotation_mlp": {"in_features": 1062, "hidden_features": 100, "out_features": 4, "activation": "relu"},
        "translation_mlp": {"in_features": 1062, "hidden_features": 100, "out_features": 3, "activation": "relu"}
    }
}

try:
    model = build_model(model_cfg)
    model.train()  # 학습 모드 설정
    print("모델이 성공적으로 생성되었습니다.")
except Exception as e:
    print(f"모델 생성 중 에러 발생: {e}")
    exit()

# 검증 코드
try:
    for i, data in enumerate(dataloader):
        if i >= 5:  # 예시로 5개의 샘플만 검증
            break

        print(f"\nSample {i+1} 검증:")

        # 데이터가 올바르게 로드되었는지 확인
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f" - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f" - {key}: type={type(value)}")

        # 1. t-1 시점 특징 추출
        prev_coord = data['prev_coord'].squeeze(0)  # 배치 차원 제거
        prev_strength = data['prev_strength'].squeeze(0)
        print(f" - t-1 데이터 (prev_coord, prev_strength) shapes: {prev_coord.shape}, {prev_strength.shape}")

        point_t1 = Point({'coord': prev_coord, 'strength': prev_strength})
        point_t1 = model.backbone(point_t1)
        feat_t1 = point_t1.feat if isinstance(point_t1, Point) else point_t1
        print(f" - t-1 특징 (feat_t1) shape: {feat_t1.shape}")

        # 2. t 시점 특징 추출
        coord = data['coord'].squeeze(0)  # 배치 차원 제거
        strength = data['strength'].squeeze(0)
        print(f" - t 데이터 (coord, strength) shapes: {coord.shape}, {strength.shape}")

        point_t = Point({'coord': coord, 'strength': strength})
        point_t = model.backbone(point_t)
        feat_t = point_t.feat if isinstance(point_t, Point) else point_t
        print(f" - t 특징 (feat_t) shape: {feat_t.shape}")

        # 3. 특징 결합 확인
        combined_features = torch.cat([feat_t, feat_t1], dim=1)
        print(f" - 결합된 특징 (combined_features) shape: {combined_features.shape}")
except Exception as e:
    print(f"검증 중 에러 발생: {e}")
