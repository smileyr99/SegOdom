"""
Structured3D Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
from collections.abc import Sequence

from .defaults import DefaultDataset  # 기본 데이터셋 클래스 가져오기
from .builder import DATASETS  # 데이터셋 빌더 모듈 가져오기


@DATASETS.register_module()
class Structured3DDataset(DefaultDataset):
    # Structured3D 데이터셋 클래스 정의, DefaultDataset을 상속받음

    def get_data_list(self):
        # 데이터를 불러올 파일 목록을 생성하는 메서드
        if isinstance(self.split, str):
            # split이 문자열인 경우 특정 경로 패턴에 맞는 파일 목록을 glob으로 검색
            data_list = glob.glob(
                os.path.join(self.data_root, self.split, "scene_*/room_*")
            )
        elif isinstance(self.split, Sequence):
            # split이 시퀀스인 경우 각 split 항목에 대해 glob으로 파일 목록을 검색하여 결합
            data_list = []
            for split in self.split:
                data_list += glob.glob(
                    os.path.join(self.data_root, split, "scene_*/room_*")
                )
        else:
            raise NotImplementedError  # split이 문자열이나 시퀀스가 아닌 경우 예외 발생
        return data_list  # 데이터 리스트 반환

    def get_data_name(self, idx):
        # 주어진 인덱스의 데이터 이름을 생성하는 메서드
        file_path = self.data_list[idx % len(self.data_list)]  # 인덱스에 맞는 파일 경로 가져오기
        dir_path, room_name = os.path.split(file_path)  # 파일 경로에서 디렉토리와 방 이름 분리
        scene_name = os.path.basename(dir_path)  # 디렉토리 경로에서 장면 이름 추출
        data_name = f"{scene_name}_{room_name}"  # 장면 이름과 방 이름을 결합하여 데이터 이름 생성
        return data_name  # 데이터 이름 반환
