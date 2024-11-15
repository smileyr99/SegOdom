import sys
import torch.nn as nn
import spconv.pytorch as spconv
from collections import OrderedDict
from pointcept.models.utils.structure import Point


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    포인트 클라우드 데이터를 처리하는 여러 층(레이어)들을 순서대로 쌓을 수 있게 해주는 도구
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # 1. OrderedDict로 초기화하는 경우
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        # 2. 순차적으로 모듈을 추가하는 경우
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        # 3. 키워드 인자로 모듈을 추가하는 경우
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        # 인덱스로 모듈 접근
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        # 모듈 개수 반환
        return len(self._modules)

    def add(self, module, name=None):
        # 새로운 모듈 추가
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # 1. Point 모듈 처리
            if isinstance(module, PointModule):
                input = module(input)
                
            # 2. Sparse Convolution 모듈 처리
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    # Point 객체의 경우
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    # 일반적인 경우
                    input = module(input)
                    
            # 3. 일반 PyTorch 모듈 처리
            else:
                if isinstance(input, Point):
                    # Point 객체의 경우
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    # 희소 텐서의 경우
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    # 일반적인 경우
                    input = module(input)
        return input
