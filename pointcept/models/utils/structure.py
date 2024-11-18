import torch
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None
from addict import Dict

from pointcept.models.utils.serialization import encode, decode
from pointcept.models.utils import offset2batch, batch2offset


class Point(Dict):
    """
    Point 구조체(Pointcept용)

    Pointcept에서 사용하는 Point(포인트 클라우드)는 배치된 포인트 클라우드의 다양한 속성을 포함한 딕셔너리입니다.
    이 클래스에서 정의한 주요 속성과 그 역할은 다음과 같습니다:

    필수 속성:
    - "coord": 원래의 포인트 클라우드 좌표
    - "grid_coord": 특정 그리드 크기에서의 그리드 좌표(GridSampling 관련)
    
    선택 속성:
    - "offset": 없을 경우, batch size가 1로 초기화됨
    - "batch": 없을 경우, batch size가 1로 초기화됨
    - "feat": 포인트 클라우드의 특징(feature), 모델에 입력되는 기본값
    - "grid_size": 포인트 클라우드의 그리드 크기(GridSampling 관련)
    
    직렬화 관련 속성:
    - "serialized_depth": 직렬화 깊이(2 ** depth * grid_size가 포인트 클라우드 범위를 설명)
    - "serialized_code": 직렬화 코드 리스트
    - "serialized_order": 코드에 의해 결정된 직렬화 순서
    - "serialized_inverse": 코드에 의해 결정된 역 매핑
    
    SpConv 관련 속성:
    - "sparse_shape": Sparse Conv Tensor의 Sparse 형태
    - "sparse_conv_feat": Point로부터 초기화된 SparseConvTensor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "offset"이나 "batch" 중 하나가 없을 경우, 다른 것을 기반으로 생성
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        포인트 클라우드 직렬화

        ["grid_coord" 또는 "coord"+"grid_size", "batch", "feat"]를 기반으로 수행.
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # GridSampling을 데이터 증강 과정에서 수행하지 않을 경우,
            # 아래와 같은 설정을 파이프라인에 추가 필요:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # 직렬화 큐브의 깊이(길이=2^depth)를 동적으로 계산
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth

        # 직렬화 코드의 최대 비트 길이 확인 (int64 기준 63비트 제한)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # OCNN 방식으로 depth를 16 이하로 제한
        assert depth <= 16

        # 직렬화 코드 생성 및 정렬
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        # 직렬화 순서를 임의로 섞을 경우
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # 속성 저장
        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        포인트 클라우드 희소화(Sparsification)

        포인트 클라우드 데이터를 SpConv의 SparseConvTensor로 변환.

        ["grid_coord" 또는 "coord"+"grid_size", "batch", "feat"]를 기반으로 준비.
        pad: 희소화를 위한 패딩 크기.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # GridSampling을 데이터 증강 과정에서 수행하지 않을 경우,
            # 아래와 같은 설정을 파이프라인에 추가 필요:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            # 희소화 형태 계산
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        # SpConv의 SparseConvTensor 생성
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreetization(self, depth=None, full_depth=None):
        """
        포인트 클라우드 옥트리화(Octree Generation)

        포인트 클라우드 데이터를 OCNN을 사용해 옥트리로 변환.
        ["grid_coord", "batch", "feat"]를 기반으로 수행.
        """
        assert (
            ocnn is not None
        ), "https://github.com/octree-nn/ocnn-pytorch에서 ocnn 설치 필요."
        assert {"grid_coord", "feat", "batch"}.issubset(self.keys())
        # 좌표 범위를 변경하여 옥트리화 지원
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 2
        self["depth"] = depth
        assert depth <= 16  # OCNN의 최대 depth 제한

        # [0, 2**depth] -> [0, 2] -> [-1, 1]로 변환
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        # 옥트리 생성 및 이웃 정보 구축
        octree.build_octree(point)
        octree.construct_all_neigh()
        self["octree"] = octree
