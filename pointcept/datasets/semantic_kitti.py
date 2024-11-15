"""
semantic_kitti.py
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

** 수정사항 **
- get_data_list(): 연속된 프레임 pair 반환하도록 수정
- get_data(): t, t-1 시점의 데이터와 rel_pose 로드하도록 수정

"""

import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SemanticKITTIDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError # raise: Python에서 예외를 발생시키는 키워드

        '''
        data_list에 데이터 담기
        data_list = [
            "/data/dataset/sequences/00/velodyne/000001.bin",
            "/data/dataset/sequences/00/velodyne/000002.bin",
            "/data/dataset/sequences/01/velodyne/000001.bin",
            "/data/dataset/sequences/01/velodyne/000002.bin",
            "/data/dataset/sequences/01/velodyne/000003.bin",
        ]
        '''
        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))

            # pair 구성
            for i in range(1, len(seq_files)): # 첫번째 프레임은 제외하고 시작
                frame_t = os.path.join(seq_folder, "velodyne", seq_files[i])
                frame_t1 = os.path.join(seq_folder, "velodyne", seq_files[i-1])
                data_list.append((frame_t, frame_t1))
        return data_list

    def get_data(self, idx):
        frame_t_path, frame_t1_path = self.data_list[idx % len(self.data_list)] # pair data path 가져오기
        
        # 현재 프레임(t) 데이터 로드
        with open(frame_t_path, "rb") as b:
            scan_t = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord_t = scan_t[:, :3] # (x, y, z) 좌표를 추출하여 coord 변수에 저장 (coord: 포인트 클라우드의 좌표정보만 담고 있는 배열)
        strength_t  = scan_t[:, -1].reshape([-1, 1])  # intensity값을 strength 배열에 저장 (strengh: 포인트 클라우드 intensity를 담고 있는 배열)


        # 이전 프레임(t-1) 데이터 로드
        with open(frame_t1_path, "rb") as b:
            scan_t1 = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord_t1 = scan_t1[:, :3] 
        strength_t1 = scan_t1[:, -1].reshape([-1, 1]) 

        # pose 데이터 로드 및 변화량 계산
        seq_folder = os.path.dirname(os.path.dirname(frame_t_path))
        pose_file = os.path.join(seq_folder, 'poses.txt')

        # frame number 추출 (파일명에서)
        frame_t_idx = int(os.path.basename(frame_t_path).split('.')[0])
        frame_t1_idx = int(os.path.basename(frame_t1_path).split('.')[0])

        with open(pose_file, 'r') as f:
            poses = f.readlines()
            pose_t = np.array([float(x) for x in poses[frame_t_idx].strip().split()]).reshape(3, 4)
            pose_t1 = np.array([float(x) for x in poses[frame_t1_idx].strip().split()]).reshape(3, 4)

        # 3x4 -> 4x4로 변환
        pose_t_homogeneous = np.eye(4)
        pose_t_homogeneous[:3, :4] = pose_t

        pose_t1_homogeneous = np.eye(4)
        pose_t1_homogeneous[:3, :4] = pose_t1

        # 상대적 pose 변화량 계산 (역행렬 없이 직접 계산)
        rel_pose = pose_t1_homogeneous.T @ pose_t_homogeneous

        label_file = frame_t_path.replace("velodyne", "labels").replace(".bin", ".label") # 위 data_path 리스트에서 velodyne-> labels로 .bin-> .label로 변경
        if os.path.exists(label_file): # 라벨 파일 경로가 있는지 확인 있으면 진행
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1) # 파일 a로부터 int32 형식으로 데이터를 읽어 numpy배열로 변환
                segment = np.vectorize(self.learning_map.__getitem__)( # labels파일에서 16비트 형식으로 되어 있는 라벨값을 읽어와 int32로 변환 후 1차원 배열로 저장
                    segment & 0xFFFF
                ).astype(np.int32)  # 즉 segment: 라벨값이 저장된 1차원 배열
        else: # 라벨 경로가 없을 경우
            segment = np.zeros(scan_t.shape[0]).astype(np.int32)  # 라벨값이 없을 경우 0으로 채움
        data_dict = dict(  # data_dict라는 딕셔너리 생성
            coord=coord_t,  # t; x,y,z 값
            strength=strength_t, # t; intensity
            prev_coord=coord_t1, # t-1; x,y,z 값
            prev_strength=strength_t1, # t-1; intensity
            segment=segment, # 라벨값
            rel_pose=rel_pose , # pose변화량
            name=self.get_data_name(idx), # 인덱스에 해당하는 데이터 이름 반환
        )
        return data_dict

    def get_data_name(self, idx):
        frame_t_path, frame_t1_path = self.data_list[idx % len(self.data_list)]  # 두 개의 경로를 가져옴
        sequence_name = os.path.basename(os.path.dirname(os.path.dirname(frame_t_path)))
        
        # frame number 추출 (파일명에서)
        frame_t_idx = os.path.splitext(os.path.basename(frame_t_path))[0]
        frame_t1_idx = os.path.splitext(os.path.basename(frame_t1_path))[0]
        
        # 두 프레임의 인덱스를 결합하여 페어 이름 생성
        data_name = f"{sequence_name}_{frame_t_idx}-{frame_t1_idx}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv
