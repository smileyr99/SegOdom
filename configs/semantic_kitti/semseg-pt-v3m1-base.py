_base_ = ["../_base_/default_runtime.py"]

'''
    **수정 사항**
    1. pose_head 추가
        - rotation_mlp와 translation_mlp 구성
        - in_channels와 activation function 설정

    2. criteria에 PoseLoss 추가
        - rotation_weight와 translation_weight 설정

    3. scheduler와 optimizer 설정 수정
        - pose_head에 대한 learning rate 추가
        - param_dicts에 pose_head 파라미터 설정
    
    4. evaluation metric 추가
        - rotation_error와 translation_error 설정

    5. dataset 설정 수정   
        - transform의 keys에 prev_coord, prev_strength 추가
        - Collect에 rel_pose 추가
        - train/val/test 모두에 pair 데이터 처리 추가
'''

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
mix_prob = 0   # pair 데이터에 대한 mixing은 비활성화 
empty_cache = True   # pair 데이터로 인한 메모리 관리를 위해 활성화
enable_amp = True    # 메모리 효율을 위해 유지

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=19,  # Semantic KITTI의 클래스 수
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(128, 128, 128, 128, 128),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(128, 128, 128, 128),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("SemanticKITTI"),
    ),
    # Pose Estimation을 위한 MLP head 추가
    pose_head=dict(
        type="PoseHead",
        in_features=1062,      # (512 + 19) * 2
        hidden_features=100,   # 중간 레이어 차원
        activation='relu'      # 활성화 함수
    ),
    criteria=[
        # Segmentation Loss
        dict(type="CrossEntropyLoss",
            weight=[3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704, 10.1922, 1.6155, 4.2187,
                    1.9385, 5.5455, 2.0198, 2.6261, 1.3212, 5.1102, 2.5492, 5.8585, 7.3929],
            loss_weight=1.0,
            ignore_index=-1),
        dict(type="LovaszLoss", 
            mode="multiclass", 
            loss_weight=1.0, 
            ignore_index=-1),
        # Pose Loss
        dict(type="PoseLoss",
            rotation_weight=1.0,
            translation_weight=1.0,
            static_weight=2.0,        # 정적 라벨 가중치 추가
            epsilon=1e-6,             # 수치 안정성을 위한 epsilon 추가
            static_classes=[8, 9, 10, 12, 13, 14, 15, 16, 17, 18])  # 정적 클래스 인덱스
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002, 0.001],  # [base_lr, block_lr, pose_head_lr]
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

param_dicts = [
    dict(keyword="block", lr=0.0002),     # PTv3 block
    dict(keyword="pose_head", lr=0.001)    # pose estimation head
]

# evaluation metric 추가
evaluation = dict(
    metrics=['miou'],  # segmentation metric
    pose_metrics=dict(
        rotation_error=dict(
            type="RotationError",
            threshold=5.0  # 5도 이내 오차를 성공으로 간주
        ),
        translation_error=dict(
            type="TranslationError",
            threshold=0.05  # 5cm 이내 오차를 성공으로 간주
        ),
    )
)

# dataset settings
dataset_type = "SemanticKITTIDataset"
data_root = "data/semantic_kitti"
ignore_index = -1
names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

data = dict(
    num_classes=19,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                # prev_coord, prev_strength 추가
                keys=("coord", "strength", "segment", "prev_coord", "prev_strength"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                # 모델에 필요한 모든 데이터를 지정
                keys=("coord", "grid_coord", "segment", "prev_coord", "prev_strength", "rel_pose"),
                # 모델의 입력 특징으로 사용될 데이터를 지정 (backbone에 입력될 특징들을 지정하는 부분) - prev_coord, prev_strength 추가
                feat_keys=("coord", "strength", "prev_coord", "prev_strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                # prev_coord, prev_strength 추가
                keys=("coord", "strength", "prev_coord", "prev_strength"),
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                # rel_pose, prev 관련 키 추가
                keys=("coord", "grid_coord", "index", "prev_coord", "prev_strength", "rel_pose"),
                feat_keys=("coord", "strength", "prev_coord", "prev_strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength", "prev_coord", "prev_strength")
            ),
            crop=None,
            post_transform=[
                dict(
                    type="PointClip",
                    point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    # prev 관련 키와 rel_pose 추가 
                    keys=("coord", "grid_coord", "segment", "prev_coord", "prev_strength", "rel_pose"),
                    feat_keys=("coord", "strength", "prev_coord", "prev_strength"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)