"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        # patch_size: 각 패치에 포함될 포인트의 수
        self.patch_size = patch_size
        # num_heads: 멀티헤드 어텐션의 헤드 수
        self.num_heads = num_heads
        # 상대 위치 인코딩의 경계값 계산 (3D 공간에서의 범위)
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        # 양방향 위치 인코딩을 위한 전체 범위
        self.rpe_num = 2 * self.pos_bnd + 1
        # x,y,z 3차원에 대한 위치 인코딩 테이블 생성
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        # 테이블 초기화 (정규분포 사용)
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        # 좌표값을 경계값 내로 제한하고 양수 인덱스로 변환
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # 경계값 내로 제한
            + self.pos_bnd  # 음수 인덱스를 양수로 변환
            + torch.arange(3, device=coord.device) * self.rpe_num  # x,y,z 축별 오프셋
        )
        # 위치 인코딩 테이블에서 해당 인덱스의 값을 추출
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        # 차원 재구성 및 축 방향 합산
        out = out.view(idx.shape + (-1,)).sum(3)
        # 차원 순서 변경 (N,K,K,H) -> (N,H,K,K)
        out = out.permute(0, 3, 1, 2)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,  # 입력 채널 수
        num_heads,  # 어텐션 헤드 수
        patch_size,  # 패치 크기
        qkv_bias=True,  # QKV 변환에 바이어스 사용 여부
        qk_scale=None,  # Q,K 스케일링 팩터
        attn_drop=0.0,  # 어텐션 드롭아웃 비율
        proj_drop=0.0,  # 프로젝션 드롭아웃 비율
        order_index=0,  # 시리얼화 순서 인덱스
        enable_rpe=False,  # 상대 위치 인코딩 사용 여부
        enable_flash=True,  # Flash Attention 사용 여부
        upcast_attention=True,  # 어텐션 계산시 float32 사용 여부
        upcast_softmax=True,  # softmax 계산시 float32 사용 여부
    ):
        super().__init__()
        # 채널 수가 헤드 수로 나누어 떨어져야 함
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        # 스케일링 팩터 설정 (제공되지 않은 경우 계산)
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        # Flash Attention이 활성화된 경우의 설정
        if enable_flash:
            # RPE와 Flash Attention은 함께 사용할 수 없음
            assert enable_rpe is False, "Set enable_rpe to False when enable Flash Attention"
            # Flash Attention 사용시 upcast 옵션들은 비활성화되어야 함
            assert upcast_attention is False
            assert upcast_softmax is False
            # flash_attn 라이브러리가 설치되어 있어야 함
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # Flash Attention을 사용하지 않을 때
            # patch_size는 실제 포인트 수와 최대 패치 크기 중 작은 값으로 설정
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        # QKV 변환을 위한 선형 레이어
        # channels -> channels*3 (Query, Key, Value를 한번에 계산)
        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        # 최종 출력을 위한 프로젝션 레이어
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        # RPE가 활성화된 경우 RPE 모듈 초기화
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        """상대적 위치 정보를 계산하는 메소드"""
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            # 그리드 좌표를 가져와서 패치 크기에 맞게 재구성
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            # 각 점들 간의 상대적 위치 계산
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        """패딩과 역변환 인덱스를 계산하는 메소드"""
        # 캐시 키 정의
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        
        # 캐시된 값이 없는 경우 계산
        if not all(k in point.keys() for k in [pad_key, unpad_key, cu_seqlens_key]):
            offset = point.offset
            # 각 배치의 포인트 수 계산
            bincount = offset2bincount(offset)
            # 패치 크기에 맞게 패딩된 크기 계산
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # patch_size보다 큰 포인트 그룹만 패딩
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            
            # offset 패딩 처리
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            
            # 패딩된 인덱스와 원본 인덱스 생성
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            
            # 누적 시퀀스 길이 저장을 위한 리스트
            cu_seqlens = []
            
            # 각 배치에 대해 처리
            for i in range(len(offset)):
                # unpad 인덱스 조정
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                
                # 패딩이 필요한 경우 패딩 처리
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                
                # pad 인덱스 조정
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                
                # 누적 시퀀스 길이 계산
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
                
            # 결과 저장
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        # Flash Attention이 비활성화된 경우 patch_size 조정
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        # 패딩 및 역변환 인덱스 가져오기
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        # 시리얼화된 순서에 따라 인덱스 조정
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # QKV 변환
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # QKV를 분리하고 reshape: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # 어텐션 계산
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            
            # RPE가 활성화된 경우 상대 위치 정보 추가
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
                
            # Softmax 계산
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            
            # 최종 특징 계산
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            # Flash Attention 사용하는 경우
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                # QKV를 half precision으로 변환하고 reshape
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,  # 누적 시퀀스 길이
                max_seqlen=self.patch_size,  # 최대 시퀀스 길이
                dropout_p=self.attn_drop if self.training else 0,  # 학습시에만 dropout 적용
                softmax_scale=self.scale,  # 스케일링 팩터
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)  # 원래 dtype으로 변환
            
        # inverse 인덱스를 사용하여 원래 순서로 복원
        feat = feat[inverse]

        # 최종 프로젝션과 dropout 적용
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,  # 입력 채널 수
        hidden_channels=None,  # 은닉층 채널 수 (None이면 in_channels 사용)
        out_channels=None,  # 출력 채널 수 (None이면 in_channels 사용)
        act_layer=nn.GELU,  # 활성화 함수
        drop=0.0,  # Dropout 비율
    ):
        super().__init__()
        # 채널 수 설정
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        
        # 첫 번째 완전연결층
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        # 활성화 함수
        self.act = act_layer()
        # 두 번째 완전연결층
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        # Dropout 레이어
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # MLP forward 패스
        x = self.fc1(x)  # 첫 번째 선형 변환
        x = self.act(x)  # 활성화 함수
        x = self.drop(x)  # Dropout
        x = self.fc2(x)  # 두 번째 선형 변환
        x = self.drop(x)  # Dropout
        return x


# Transformer Block 클래스 설명
class Block(PointModule):
    def __init__(
        self,
        channels,  # 입력/출력 채널 수
        num_heads,  # 어텐션 헤드 수
        patch_size=48,  # 패치 크기
        mlp_ratio=4.0,  # MLP 은닉층 크기 비율
        qkv_bias=True,  # QKV 변환에 바이어스 사용
        qk_scale=None,  # QK 스케일링 팩터
        attn_drop=0.0,  # 어텐션 dropout 비율
        proj_drop=0.0,  # 프로젝션 dropout 비율
        drop_path=0.0,  # 드롭패스 비율
        norm_layer=nn.LayerNorm,  # 정규화 레이어
        act_layer=nn.GELU,  # 활성화 함수
        pre_norm=True,  # 정규화 순서 (pre/post)
        order_index=0,  # 시리얼화 순서 인덱스
        cpe_indice_key=None,  # Conditional Position Encoding 키
        enable_rpe=False,  # 상대 위치 인코딩 사용
        enable_flash=True,  # Flash Attention 사용
        upcast_attention=True,  # 어텐션 계산 시 float32 사용
        upcast_softmax=True,  # softmax 계산 시 float32 사용
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        # Conditional Position Encoding 레이어
        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        # 첫 번째 정규화 레이어
        self.norm1 = PointSequential(norm_layer(channels))
        
        # 어텐션 레이어
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        # 두 번째 정규화 레이어
        self.norm2 = PointSequential(norm_layer(channels))
        
        # MLP 레이어
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),  # mlp_ratio로 은닉층 크기 확장
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        
        # DropPath (Stochastic Depth)
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )


    def forward(self, point: Point):
        # 첫 번째 Skip Connection (CPE)
        shortcut = point.feat
        point = self.cpe(point)  # Conditional Position Encoding 적용
        point.feat = shortcut + point.feat  # Skip Connection
        
        # 두 번째 Skip Connection (Self-Attention)
        shortcut = point.feat
        if self.pre_norm:  # Pre-normalization
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))  # Attention과 DropPath 적용
        point.feat = shortcut + point.feat  # Skip Connection
        if not self.pre_norm:  # Post-normalization
            point = self.norm1(point)

        # 세 번째 Skip Connection (MLP)
        shortcut = point.feat
        if self.pre_norm:  # Pre-normalization
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))  # MLP와 DropPath 적용
        point.feat = shortcut + point.feat  # Skip Connection
        if not self.pre_norm:  # Post-normalization
            point = self.norm2(point)
            
        # 스파스 컨볼루션 특징 업데이트
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,  # 입력 채널 수
        out_channels,  # 출력 채널 수
        stride=2,  # 풀링 스트라이드 (2,4,8만 가능)
        norm_layer=None,  # 정규화 레이어
        act_layer=None,  # 활성화 함수
        reduce="max",  # 풀링 방식 (max, mean, min, sum)
        shuffle_orders=True,  # 순서 섞기 여부
        traceable=True,  # 부모와 클러스터 추적 여부
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 스트라이드가 2의 거듭제곱인지 확인 (2, 4, 8)
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        
        # 풀링 방식 검증
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        # 채널 변환을 위한 선형 레이어
        self.proj = nn.Linear(in_channels, out_channels)
        # 선택적 정규화 레이어
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        # 선택적 활성화 레이어
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        # 풀링 깊이 계산 (스트라이드에 따른 비트 이동량)
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
            
        # 필요한 키들이 있는지 확인
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(point.keys()), "Run point.serialization() first"

        # 코드를 풀링 깊이만큼 우측 시프트
        code = point.serialized_code >> pooling_depth * 3
        # 유니크한 코드 값과 클러스터링 정보 얻기
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        
        # 클러스터로 정렬된 포인트 인덱스
        _, indices = torch.sort(cluster)
        # 정렬된 포인트의 인덱스 포인터
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # 각 클러스터의 대표 인덱스
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        # 다운샘플링된 코드와 순서 생성
        code = code[:, head_indices]
        order = torch.argsort(code)  # 코드 기반 정렬
        # 역순서 인덱스 생성
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        # 순서 섞기 옵션이 활성화된 경우
        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # 새로운 포인트 딕셔너리 생성
        point_dict = Dict(
            # 특징 벡터 풀링 (선택된 reduce 방식 사용)
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            # 좌표 평균값 계산
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            # 그리드 좌표 다운샘플링
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        # 조건과 컨텍스트 정보 유지
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        # 추적 정보 저장 (활성화된 경우)
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
            
        # 새로운 Point 객체 생성
        point = Point(point_dict)
        # 정규화 적용 (있는 경우)
        if self.norm is not None:
            point = self.norm(point)
        # 활성화 함수 적용 (있는 경우)
        if self.act is not None:
            point = self.act(point)
        # 스파스 텐서 변환
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,      # 입력 채널 수
        skip_channels,    # 스킵 커넥션의 채널 수
        out_channels,     # 출력 채널 수
        norm_layer=None,  # 정규화 레이어
        act_layer=None,   # 활성화 함수
        traceable=False,  # 추적 여부
    ):
        super().__init__()
        # 메인 특징 프로젝션
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        # 스킵 커넥션 특징 프로젝션
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        # 정규화 레이어 추가 (있는 경우)
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        # 활성화 레이어 추가 (있는 경우)
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        # pooling_parent와 pooling_inverse가 있는지 확인
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        # 부모 포인트와 역인덱스 추출
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        
        # 현재 레벨과 스킵 커넥션 특징 프로젝션
        point = self.proj(point)
        parent = self.proj_skip(parent)
        # 특징 결합 (additive skip connection)
        parent.feat = parent.feat + point.feat[inverse]

        # 추적 정보 저장 (활성화된 경우)
        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,      # 입력 채널 수
        embed_channels,   # 임베딩 채널 수
        norm_layer=None,  # 정규화 레이어
        act_layer=None,   # 활성화 함수
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # 초기 특징 추출을 위한 스파스 컨볼루션 레이어
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,    # 5x5x5 커널
                padding=1,        # 패딩 크기
                bias=False,       # 바이어스 미사용
                indice_key="stem" # 인덱스 키
            )
        )
        # 정규화 레이어 추가 (있는 경우)
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        # 활성화 레이어 추가 (있는 경우)
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,            # 입력 채널 수 (x,y,z + RGB 등)
        order=("z", "z-trans"),   # 시리얼화 순서
        stride=(2, 2, 2, 2),      # 각 스테이지의 스트라이드
        # 인코더 구성
        enc_depths=(2, 2, 2, 6, 2),         # 각 스테이지의 블록 수
        enc_channels=(32, 64, 128, 256, 512), # 각 스테이지의 채널 수
        enc_num_head=(2, 4, 8, 16, 32),     # 각 스테이지의 어텐션 헤드 수
        enc_patch_size=(48, 48, 48, 48, 48), # 각 스테이지의 패치 크기
        # 디코더 구성
        dec_depths=(2, 2, 2, 2),           # 디코더 각 스테이지의 블록 수
        dec_channels=(64, 64, 128, 256),   # 디코더 각 스테이지의 채널 수
        dec_num_head=(4, 4, 8, 16),        # 디코더 각 스테이지의 어텐션 헤드 수
        dec_patch_size=(48, 48, 48, 48),   # 디코더 각 스테이지의 패치 크기
        # 모델 구성 파라미터
        mlp_ratio=4,             # MLP 확장 비율
        qkv_bias=True,           # QKV 변환에 바이어스 사용
        qk_scale=None,           # QK 스케일링 팩터
        attn_drop=0.0,           # 어텐션 드롭아웃 비율
        proj_drop=0.0,           # 프로젝션 드롭아웃 비율
        drop_path=0.3,           # 드롭패스 비율
        pre_norm=True,           # 선행 정규화 사용
        shuffle_orders=True,     # 순서 섞기 사용
        # 추가 기능 활성화
        enable_rpe=False,        # 상대 위치 인코딩
        enable_flash=True,       # Flash Attention
        upcast_attention=False,  # 어텐션 계산시 float32 사용
        upcast_softmax=False,    # softmax 계산시 float32 사용
        cls_mode=False,          # 분류 모드 사용
        # PDNorm 관련 설정
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        # 스테이지 수 설정
        self.num_stages = len(enc_depths)
        # 시리얼화 순서 설정
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        # 구성 검증
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        # 분류 모드가 아닌 경우 디코더 구성 검증
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # 정규화 레이어 설정
        if pdnorm_bn:
            # PDNorm을 사용하는 배치 정규화
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            # 일반 배치 정규화
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            
        if pdnorm_ln:
            # PDNorm을 사용하는 레이어 정규화
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            # 일반 레이어 정규화
            ln_layer = nn.LayerNorm
            
        # 활성화 함수 설정
        act_layer = nn.GELU

        # 임베딩 레이어 초기화
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # 인코더 초기화
        # 드롭패스 비율을 선형적으로 분배
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        
        # 각 인코더 스테이지 구성
        for s in range(self.num_stages):
            # 현재 스테이지의 드롭패스 비율 추출
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            
            # 첫 스테이지가 아니면 풀링 레이어 추가
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
                
            # 현재 스테이지의 트랜스포머 블록 추가
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            # 현재 스테이지를 인코더에 추가
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        # 분류 모드가 아닌 경우에만 디코더 초기화
        if not self.cls_mode:
            # 드롭패스 비율을 선형적으로 분배
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            # 디코더 채널에 인코더의 마지막 채널 추가
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            
            # 역순으로 디코더 스테이지 구성 (스킵 커넥션을 위해)
            for s in reversed(range(self.num_stages - 1)):
                # 현재 스테이지의 드롭패스 비율 추출
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                # 드롭패스 순서 역전
                dec_drop_path_.reverse()
                
                dec = PointSequential()
                # 언풀링 레이어 추가 (업샘플링)
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],  # 인코더의 대응하는 레이어와 연결
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                
                # 현재 스테이지의 트랜스포머 블록 추가
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                # 현재 스테이지를 디코더에 추가
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        # 데이터를 Point 객체로 변환
        point = Point(data_dict)
        
        # 포인트 클라우드 시리얼화 수행
        # (포인트들을 특정 순서로 정렬하고 구조화)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        
        # 스파스 텐서로 변환
        point.sparsify()

        # 임베딩 레이어 통과
        point = self.embedding(point)
        
        # 인코더 통과
        point = self.enc(point)
        
        # 분류 모드가 아닌 경우 디코더 통과
        if not self.cls_mode:
            point = self.dec(point)
        # 분류 모드인 경우 (주석 처리된 코드)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
            
        return point
