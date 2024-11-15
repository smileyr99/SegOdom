"""
default.py

Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

** 수정사항 **
- VALID_ASSETS에 추가: 'prev_coord', 'prev_strength', 'rel_pose'
- data_dict 형식 수정
- prepare_train_data(), prepare_test_data() 수정하여 pair 데이터 처리

"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger  # 로그 생성 함수 가져오기
from pointcept.utils.cache import shared_dict  # 캐시 공유를 위한 함수 가져오기

from .builder import DATASETS, build_dataset  # 데이터셋 생성 및 빌더 가져오기
from .transform import Compose, TRANSFORMS  # 변환 함수 가져오기


@DATASETS.register_module()
class DefaultDataset(Dataset):
    VALID_ASSETS = [
        "coord",     # 좌표
        "color",     # 색상
        "normal",    # 법선 벡터
        "strength",  # intensity
        "segment",   # label
        "instance",  # 인스턴스
        "prev_coord",    # 이전 프레임 좌표 (추가)
        "prev_strength", # 이전 프레임 강도 (추가)
        "rel_pose",      # 자세 변화량
        "pose"
    ]

    def __init__(
        self,
        split="train",  # 데이터셋 분할 설정 (train, val, test 등)
        data_root="data/dataset",  # 데이터 경로
        transform=None,  # 변환 함수
        test_mode=False,  # 테스트 모드 설정
        test_cfg=None,  # 테스트 설정
        cache=False,  # 캐시 사용 여부
        ignore_index=-1,  # 무시할 인덱스 값
        loop=1,  # 반복 횟수 설정
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)  # 여러 변환을 하나로 결합
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # 테스트 모드일 때는 루프를 1로 강제 설정
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)  # 테스트 시 복셀화 설정
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )  # 테스트 시 크롭 설정
            self.post_transform = Compose(self.test_cfg.post_transform)  # 후처리 변환 설정
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]  # 데이터 증강 변환 설정

        self.data_list = self.get_data_list()  # 데이터 파일 목록 생성
        logger = get_root_logger()  # 로거 설정
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        # split이 문자열이면, 해당 폴더의 모든 파일을 리스트로 만듦
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*"))
        # split이 시퀀스면, 각 split 폴더의 모든 파일을 리스트로 만듦
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        else:
            raise NotImplementedError  # 구현되지 않은 경우 예외 발생
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]  # 데이터 경로 설정
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)  # 캐시된 데이터 반환

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):  # .npy 확장자가 아닌 파일 무시
                continue
            if asset[:-4] not in self.VALID_ASSETS:  # 유효 자산이 아니면 무시
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))  # 데이터 로드하여 딕셔너리에 추가
        data_dict["name"] = name

        # 데이터 타입 변환
        # 현재 프레임 데이터
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)  # 좌표 데이터 형 변환
        if "strength" in data_dict.keys():
            data_dict["strength"] = data_dict["strength"].astype(np.float32) # intensity 데이터 형 변환

        # 이전 프레임 데이터
        if "prev_coord" in data_dict.keys():
            data_dict["prev_coord"] = data_dict["prev_coord"].astype(np.float32)
        if "prev_strength" in data_dict.keys():
            data_dict["prev_strength"] = data_dict["prev_strength"].astype(np.float32)
        
        # 상대 포즈 데이터
        if "rel_pose" in data_dict.keys():
            data_dict["rel_pose"] = data_dict["rel_pose"].astype(np.float32)

        # 기존 데이터 타입 변환
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)  # 색상 데이터 형 변환

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)  # 법선 벡터 데이터 형 변환

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)  # 세그먼트 데이터 형 변환
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )  # 세그먼트 데이터가 없으면 -1로 채움

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)  # 인스턴스 데이터 형 변환
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )  # 인스턴스 데이터가 없으면 -1로 채움
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])  # 데이터 이름 반환

    def prepare_train_data(self, idx):
        # 데이터 로드 및 변환 적용
        data_dict = self.get_data(idx)

        # transform 적용을 위해 데이터 분리
        data_dict_t = {
            'coord': data_dict['coord'],
            'strength': data_dict['strength'],
            'segment': data_dict['segment'] 
        }

        data_dict_t1 = {
            'coord': data_dict['prev_coord'],
            'strength': data_dict['prev_strength']
        }
        
        # 각각 transform 적용
        data_dict_t = self.transform(data_dict_t)
        data_dict_t1 = self.transform(data_dict_t1)
        
        # 결과 합치기 (rel_pose는 transform 적용 없이 유지)
        data_dict.update({
            'coord': data_dict_t['coord'],
            'strength': data_dict_t['strength'],
            'segment': data_dict_t['segment'],
            'prev_coord': data_dict_t1['coord'],
            'prev_strength': data_dict_t1['strength'],
            'rel_pose': data_dict['rel_pose']  
        })
   
        return data_dict

    def prepare_test_data(self, idx):
        # 데이터 로드 및 변환 적용
        data_dict = self.get_data(idx)
        
        # Transform 적용을 위해 데이터 분리
        data_dict_t = {
            'coord': data_dict['coord'],
            'strength': data_dict['strength'],
            'segment': data_dict['segment']
        }
        
        data_dict_t1 = {
            'coord': data_dict['prev_coord'],
            'strength': data_dict['prev_strength']
        }

        # 각각 transform 적용
        data_dict_t = self.transform(data_dict_t)
        data_dict_t1 = self.transform(data_dict_t1)
        
        # 결과 처리
        result_dict = dict(
            segment=data_dict_t.pop("segment"),
            name=data_dict.pop("name")
        )
        
        # pair 데이터 업데이트 (rel_pose는 transform 적용 없이 유지)
        data_dict.update({
            'coord': data_dict_t['coord'],
            'strength': data_dict_t['strength'],
            'prev_coord': data_dict_t1['coord'],
            'prev_strength': data_dict_t1['strength'],
            'rel_pose': data_dict['rel_pose']  # 이름 변경
        })

        # result_dict 생성 및 segment, name 추출
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        # Test mode에서의 추가 처리
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))  # 데이터 증강

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)  # 복셀화 적용
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)  # 크롭 적용
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])  # 후처리 변환 적용
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)  # 테스트 데이터 준비
        else:
            return self.prepare_train_data(idx)  # 훈련 데이터 준비

    def __len__(self):
        return len(self.data_list) * self.loop  # 전체 데이터셋 길이 반환


@DATASETS.register_module()
class ConcatDataset(Dataset):
    def __init__(self, datasets, loop=1):
        super(ConcatDataset, self).__init__()
        self.datasets = [build_dataset(dataset) for dataset in datasets]  # 각 데이터셋 빌드
        self.loop = loop  # 반복 횟수 설정
        self.data_list = self.get_data_list()  # 데이터 목록 생성
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in the concat set.".format(
                len(self.data_list), self.loop
            )
        )

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i])) * i, np.arange(len(self.datasets[i]))
                )
            )  # 데이터셋 인덱스와 데이터 인덱스 조합하여 데이터 목록 생성
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]  # 데이터셋 인덱스 및 데이터 인덱스 가져오기
        return self.datasets[dataset_idx][data_idx]  # 데이터 반환

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)  # 데이터 이름 반환

    def __getitem__(self, idx):
        return self.get_data(idx)  # 데이터 반환

    def __len__(self):
        return len(self.data_list) * self.loop  # 데이터셋 전체 길이 반환
