# import torch
# from pointcept.datasets.builder import build_dataset

# def test_data_loading(dataset_cfg, num_samples=5):
#     """
#     데이터 로드 검증 함수. 설정된 데이터셋을 불러오고,
#     각 필드가 예상대로 로드되는지 확인합니다.
    
#     Args:
#         dataset_cfg (dict): 데이터셋 설정 구성.
#         num_samples (int): 검증할 샘플 수.
#     """
#     dataset = build_dataset(dataset_cfg)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#     print("데이터 로드 검증을 시작합니다...")
#     for i, data in enumerate(dataloader):
#         if i >= num_samples:
#             break

#         print(f"\nSample {i+1} 검증:")

#         # 각 데이터 필드의 크기와 타입을 출력합니다.
#         for key, value in data.items():
#             if isinstance(value, torch.Tensor):
#                 print(f" - {key}: shape={value.shape}, dtype={value.dtype}")
#             else:
#                 print(f" - {key}: type={type(value)}")

#         # 특정 필드가 존재하는지 확인합니다 (예: coord, strength, segment, prev_coord, prev_strength, rel_pose).
#         required_fields = ["coord", "strength", "segment", "prev_coord", "prev_strength", "rel_pose"]
#         for field in required_fields:
#             if field not in data:
#                 print(f"필드 '{field}'이(가) 존재하지 않습니다. 확인이 필요합니다.")
#             else:
#                 print(f"필드 '{field}'이(가) 정상적으로 로드되었습니다.")

#         # rel_pose 값을 확인합니다.
#         if "rel_pose" in data:
#             print(" - rel_pose 값 확인:")
#             print(data["rel_pose"])  # rel_pose 값을 출력합니다.
    
#     print("\n데이터 로드 검증이 완료되었습니다.")

# # 데이터셋 구성 설정 예시
# dataset_cfg = {
#     "type": "SemanticKITTIDataset",
#     "split": "train",
#     "data_root": "data/semantic_kitti",
#     "transform": [
#         {"type": "ToTensor"},
#     ],
#     "test_mode": False
# }

# # 데이터 로드 검증 함수 호출
# test_data_loading(dataset_cfg)


import torch
from pointcept.datasets.builder import build_dataset

def test_data_loading(dataset_cfg, num_samples=5):
    dataset = build_dataset(dataset_cfg)
    
    # 데이터 리스트가 잘 만들어졌는지 확인
    print("\n데이터 리스트가 잘 만들어졌는지 확인합니다...")
    for i in range(num_samples):
        frame_pair = dataset.data_list[i]
        print(f"Sample {i+1} pair: {frame_pair}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print("데이터 로드 검증을 시작합니다...")
    for i, data in enumerate(dataloader):
        if i >= num_samples:
            break

        print(f"\nSample {i+1} 검증:")

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f" - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f" - {key}: type={type(value)}")

        required_fields = ["coord", "strength", "segment", "prev_coord", "prev_strength", "rel_pose"]
        for field in required_fields:
            if field not in data:
                print(f"필드 '{field}'이(가) 존재하지 않습니다. 확인이 필요합니다.")
            else:
                print(f"필드 '{field}'이(가) 정상적으로 로드되었습니다.")

        if "rel_pose" in data:
            print(" - rel_pose 값 확인:")
            print(data["rel_pose"])
    
    print("\n데이터 로드 검증이 완료되었습니다.")

# 데이터셋 구성 설정 예시
dataset_cfg = {
    "type": "SemanticKITTIDataset",
    "split": "train",
    "data_root": "data/semantic_kitti",
    "transform": [
        {"type": "ToTensor"},
    ],
    "test_mode": False
}

# 데이터 로드 검증 함수 호출
test_data_loading(dataset_cfg)
