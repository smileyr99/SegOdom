"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial

# Python 버전이 3.10 이상인 경우, collections 모듈의 Iterator를 collections.abc에서 가져옵니다.
if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

# pointcept의 기본 설정을 가져옵니다.
from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

# 훈련기를 등록하는 레지스트리
TRAINERS = Registry("trainers")


# TrainerBase 클래스 정의
class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []  # 훈련 중 실행할 훅 목록
        self.epoch = 0  # 현재 에포크 번호
        self.start_epoch = 0  # 훈련 시작 에포크
        self.max_epoch = 0  # 훈련할 최대 에포크 수
        self.max_iter = 0  # 최대 반복 횟수
        self.comm_info = dict()  # 훈련에 필요한 통신 정보
        self.data_iterator: Iterator = enumerate([])  # 데이터 반복자
        self.storage: EventStorage  # 이벤트 저장소
        self.writer: SummaryWriter  # 로그를 기록할 SummaryWriter

    # 훈련 중 사용할 훅을 등록
    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)  # 훅을 빌드합니다.
        for h in hooks:
            assert isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)  # Trainer를 약한 참조로 저장하여 메모리 누수를 방지
        self.hooks.extend(hooks)

    # 훈련 루프
    def train(self):
        with EventStorage() as self.storage:
            # => 훈련 시작 전
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => 에포크 시작 전
                self.before_epoch()
                # => 에포크 실행
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => 스텝 시작 전
                    self.before_step()
                    # => 스텝 실행
                    self.run_step()
                    # => 스텝 종료 후
                    self.after_step()
                # => 에포크 종료 후
                self.after_epoch()
            # => 훈련 종료 후
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()  # 각 훅의 훈련 시작 전 처리

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()  # 각 훅의 에포크 시작 전 처리

    def before_step(self):
        for h in self.hooks:
            h.before_step()  # 각 훅의 스텝 시작 전 처리

    def run_step(self):
        raise NotImplementedError  # 스텝 실행은 하위 클래스에서 구현

    def after_step(self):
        for h in self.hooks:
            h.after_step()  # 각 훅의 스텝 종료 후 처리

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()  # 각 훅의 에포크 종료 후 처리
        self.storage.reset_histories()  # 저장된 기록 초기화

    def after_train(self):
        comm.synchronize()  # GPU 동기화
        for h in self.hooks:
            h.after_train()  # 각 훅의 훈련 종료 후 처리
        if comm.is_main_process():
            self.writer.close()  # 메인 프로세스에서만 writer 종료


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch  # 최대 에포크 수를 설정
        self.best_metric_value = -torch.inf  # 최고의 성능 값을 초기화
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Config 로드 중 ...")
        self.cfg = cfg
        self.logger.info(f"저장 경로: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> 모델 빌드 중 ...")
        self.model = self.build_model()  # 모델을 빌드
        self.logger.info("=> Writer 빌드 중 ...")
        self.writer = self.build_writer()  # Writer를 빌드
        self.logger.info("=> 훈련 데이터셋 및 로더 빌드 중 ...")
        self.train_loader = self.build_train_loader()  # 훈련 데이터 로더 빌드
        self.logger.info("=> 검증 데이터셋 및 로더 빌드 중 ...")
        self.val_loader = self.build_val_loader()  # 검증 데이터 로더 빌드
        self.logger.info("=> 옵티마이저, 스케줄러, 스케일러(amp) 빌드 중 ...")
        self.optimizer = self.build_optimizer()  # 옵티마이저 빌드
        self.scheduler = self.build_scheduler()  # 스케줄러 빌드
        self.scaler = self.build_scaler()  # 스케일러 빌드
        self.logger.info("=> 훅 빌드 중 ...")
        self.register_hooks(self.cfg.hooks)  # 훅 등록

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => 훈련 시작 전
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> 훈련 시작 >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => 에포크 시작 전
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)  # 분산 훈련 시 에포크 설정
                self.model.train()  # 모델을 훈련 모드로 설정
                self.data_iterator = enumerate(self.train_loader)  # 데이터 반복자 설정
                self.before_epoch()
                # => 에포크 실행
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => 스텝 시작 전
                    self.before_step()
                    # => 스텝 실행
                    self.run_step()
                    # => 스텝 종료 후
                    self.after_step()
                # => 에포크 종료 후
                self.after_epoch()
            # => 훈련 종료 후
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]  # 입력 데이터 가져오기
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)  # 데이터를 GPU로 전송
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):  # AMP를 이용한 자동 혼합 정밀도 활성화
            output_dict = self.model(input_dict)  # 모델을 통해 출력 생성
            loss = output_dict["loss"]  # 손실 값 추출
        self.optimizer.zero_grad()  # 옵티마이저 초기화
        if self.cfg.enable_amp:  # AMP 활성화 여부에 따라 스케일링하여 역전파
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:  # AMP가 비활성화된 경우 일반적인 역전파와 옵티마이저 스텝
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()  # GPU 메모리 캐시 비우기
        self.comm_info["model_output_dict"] = output_dict  # 모델 출력 저장

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()  # 각 훅의 에포크 종료 후 처리
        self.storage.reset_histories()  # 저장된 기록 초기화
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()  # 에포크당 캐시 비우기 설정 시 GPU 캐시 비우기

    def build_model(self):
        model = build_model(self.cfg.model)  # 모델을 생성합니다.
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 동기화 배치 정규화 적용
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"모델 매개변수 수: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None  # 메인 프로세스에만 Writer 생성
        self.logger.info(f"Tensorboard Writer 로그 디렉토리: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)  # 훈련 데이터셋 생성

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)  # 분산 샘플러 생성
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)  # 옵티마이저 생성

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch  # 총 스케줄러 스텝 수 계산
        return build_scheduler(self.cfg.scheduler, self.optimizer)  # 스케줄러 생성

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None  # AMP 활성화 시 스케일러 생성
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)  # 훈련 데이터셋 생성
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)  # 에포크당 반복 횟수 설정
        return train_loader
