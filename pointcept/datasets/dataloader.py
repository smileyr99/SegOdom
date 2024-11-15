### dataloader.py###

from functools import partial
import weakref
import torch
import torch.utils.data

import pointcept.utils.comm as comm  # 통신 관련 유틸리티 함수 가져오기
from pointcept.datasets.utils import point_collate_fn  # 데이터 병합 함수 가져오기
from pointcept.datasets import ConcatDataset  # 여러 데이터셋을 합치는 클래스
from pointcept.utils.env import set_seed  # 시드 설정 함수 가져오기


class MultiDatasetDummySampler:
    def __init__(self):
        self.dataloader = None  # 데이터 로더를 저장할 변수 초기화

    def set_epoch(self, epoch):
        # 여러 GPU가 있을 때 에폭마다 샘플러의 에폭을 설정
        if comm.get_world_size() > 1:
            for dataloader in self.dataloader.dataloaders:
                dataloader.sampler.set_epoch(epoch)  # 각 데이터 로더의 샘플러에 에폭 설정
        return


class MultiDatasetDataloader:
    """
    여러 데이터셋의 데이터 로더. 같은 데이터셋에서 배치 데이터를 로드하고 각 하위 데이터셋의 루프 비율로 비율을 섞어준다.
    전체 길이는 메인 데이터셋(첫 번째 데이터셋)과 concat 데이터셋의 루프에 의해 결정된다.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,  # 결합된 데이터셋을 받음
        batch_size_per_gpu: int,  # GPU당 배치 크기 설정
        num_worker_per_gpu: int,  # GPU당 워커(worker) 수 설정
        mix_prob=0,  # 데이터셋 혼합 확률 설정
        seed=None,  # 시드 값 설정
    ):
        self.datasets = concat_dataset.datasets  # 데이터셋 리스트 가져오기
        self.ratios = [dataset.loop for dataset in self.datasets]  # 각 데이터셋의 루프 비율 설정
        # 데이터 루프를 재설정하여 비율로 사용, 실제 루프는 1로 설정
        for dataset in self.datasets:
            dataset.loop = 1
        # 메인 데이터셋의 루프를 결합 데이터셋의 루프에 맞춰 설정
        self.datasets[0].loop = concat_dataset.loop
        # 서브 데이터 로더 빌드
        num_workers = num_worker_per_gpu // len(self.datasets)  # 각 데이터셋에 대해 워커 수 분배
        self.dataloaders = []
        for dataset_id, dataset in enumerate(self.datasets):
            if comm.get_world_size() > 1:
                # 여러 GPU가 있을 경우 분산 샘플러 생성
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None

            # 워커 초기화 함수 설정 (시드가 있는 경우)
            init_fn = (
                partial(
                    self._worker_init_fn,
                    dataset_id=dataset_id,
                    num_workers=num_workers,
                    num_datasets=len(self.datasets),
                    rank=comm.get_rank(),
                    seed=seed,
                )
                if seed is not None
                else None
            )
            self.dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size_per_gpu,
                    shuffle=(sampler is None),  # 샘플러가 없을 때만 셔플
                    num_workers=num_worker_per_gpu,
                    sampler=sampler,  # 샘플러 설정
                    collate_fn=partial(point_collate_fn, mix_prob=mix_prob),  # 데이터 병합 함수 설정
                    pin_memory=True,  # 고정 메모리 옵션 활성화
                    worker_init_fn=init_fn,  # 워커 초기화 함수 설정
                    drop_last=True,  # 마지막 불완전한 배치 드롭
                    persistent_workers=True,  # 워커가 지속적으로 실행되도록 설정
                )
            )
        self.sampler = MultiDatasetDummySampler()  # 더미 샘플러 생성
        self.sampler.dataloader = weakref.proxy(self)  # 데이터 로더의 약한 참조를 샘플러에 연결

    def __iter__(self):
        # 각 데이터 로더에 대한 반복자를 생성
        iterator = [iter(dataloader) for dataloader in self.dataloaders]
        while True:
            for i in range(len(self.ratios)):  # 데이터셋 비율에 따라 반복
                for _ in range(self.ratios[i]):
                    try:
                        batch = next(iterator[i])  # 배치 로드
                    except StopIteration:
                        if i == 0:  # 메인 데이터셋이 끝나면 종료
                            return
                        else:  # 다른 데이터셋은 반복기 재설정
                            iterator[i] = iter(self.dataloaders[i])
                            batch = next(iterator[i])
                    yield batch  # 배치 반환

    def __len__(self):
        main_data_loader_length = len(self.dataloaders[0])  # 메인 데이터 로더 길이 가져오기
        # 메인 데이터 로더 길이를 비율로 나눈 후 비율 합에 맞춘 전체 길이 반환
        return (
            main_data_loader_length // self.ratios[0] * sum(self.ratios)
            + main_data_loader_length % self.ratios[0]
        )

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, dataset_id, num_datasets, rank, seed):
        # 워커 시드 계산 및 설정
        worker_seed = (
            num_workers * num_datasets * rank
            + num_workers * dataset_id
            + worker_id
            + seed
        )
        set_seed(worker_seed)  # 시드 설정
