# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from functools import partial

from .misc import is_seq_of


def build_from_cfg(cfg, registry, default_args=None):
    """설정 딕셔너리를 사용해 모듈을 생성하는 함수.

    Args:
        cfg (dict): 설정 딕셔너리로, 최소 "type" 키를 포함해야 함.
        registry (:obj:`Registry`): 모듈을 검색할 레지스트리.
        default_args (dict, optional): 기본 초기화 인자.

    Returns:
        object: 생성된 객체.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg는 딕셔너리여야 합니다, 그러나 {type(cfg)}를 받았습니다.")  # cfg가 딕셔너리가 아니면 오류 발생
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` 또는 `default_args`에 "type" 키가 있어야 합니다, '
                f"그러나 {cfg}\n{default_args}를 받았습니다."
            )  # type이 없으면 오류 발생
    if not isinstance(registry, Registry):
        raise TypeError(
            "registry는 mmcv.Registry 객체여야 합니다, " f"그러나 {type(registry)}를 받았습니다."
        )  # registry가 Registry 객체인지 확인
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            "default_args는 딕셔너리 또는 None이어야 합니다, " f"그러나 {type(default_args)}를 받았습니다."
        )  # default_args가 None 또는 dict인지 확인

    args = cfg.copy()  # 설정 딕셔너리를 복사하여 인자로 사용

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)  # 기본 인자가 없을 경우 default_args를 설정

    obj_type = args.pop("type")  # type 키를 사용하여 객체의 클래스를 가져옴
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type}는 {registry.name} 레지스트리에 없습니다")  # 레지스트리에 type이 없는 경우 오류
    elif inspect.isclass(obj_type):
        obj_cls = obj_type  # obj_type이 클래스인 경우 그대로 사용
    else:
        raise TypeError(f"type은 문자열 또는 유효한 타입이어야 합니다, 그러나 {type(obj_type)}를 받았습니다.")
    try:
        return obj_cls(**args)  # 인자를 사용하여 객체를 생성 후 반환
    except Exception as e:
        # 일반적인 TypeError는 클래스 이름을 출력하지 않으므로 예외를 수정하여 발생시킴
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """문자열을 클래스에 매핑하는 레지스트리.

    레지스트리에 등록된 객체는 해당 객체를 생성할 수 있습니다.
    예제:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))

    고급 사용법은 다음을 참조하십시오:
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html

    Args:
        name (str): 레지스트리 이름.
        build_func(func, optional): 레지스트리에서 인스턴스를 생성하는 빌드 함수. 기본값은 `build_from_cfg` 사용.
        parent (Registry, optional): 상위 레지스트리. 부모 레지스트리에서 하위 레지스트리에 등록된 클래스를 생성할 수 있음.
        scope (str, optional): 레지스트리의 범위로, 하위 레지스트리를 검색하는 키 역할을 함.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name  # 레지스트리 이름 설정
        self._module_dict = dict()  # 모듈을 저장할 딕셔너리 초기화
        self._children = dict()  # 하위 레지스트리 저장할 딕셔너리 초기화
        self._scope = self.infer_scope() if scope is None else scope  # 범위 설정

        # self.build_func은 아래 우선순위에 따라 설정됨:
        # 1. 지정된 build_func
        # 2. 부모 레지스트리의 build_func
        # 3. 기본 build_from_cfg 사용
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func  # 부모의 build_func 사용
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)  # 부모가 Registry인지 확인
            parent._add_children(self)  # 부모에 현재 레지스트리를 하위로 추가
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)  # 레지스트리 내 모듈 수 반환

    def __contains__(self, key):
        return self.get(key) is not None  # 주어진 키가 레지스트리에 있는지 확인

    def __repr__(self):
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str  # 레지스트리의 문자열 표현 반환

    @staticmethod
    def infer_scope():
        """레지스트리의 범위를 추론하는 함수.

        레지스트리가 정의된 패키지의 이름을 반환합니다.

        예제:
            # mmdet/models/backbone/resnet.py 파일 내
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            ResNet 클래스의 범위는 "mmdet"이 됩니다.


        Returns:
            scope (str): 추론된 범위 이름.
        """
        # inspect.stack()으로 호출된 위치를 추적하며, index-2는 infer_scope()가 호출된 프레임을 의미
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]  # 첫 번째 부분을 반환하여 범위 이름으로 사용

    @staticmethod
    def split_scope_key(key):
        """범위와 키를 분리하는 함수.

        첫 번째 범위를 키에서 분리합니다.

        예제:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            scope (str, None): 첫 번째 범위.
            key (str): 남은 키.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]  # 범위와 나머지 키 반환
        else:
            return None, key  # 범위가 없는 경우 None과 키 반환

    @property
    def name(self):
        return self._name  # 레지스트리 이름 반환

    @property
    def scope(self):
        return self._scope  # 레지스트리 범위 반환

    @property
    def module_dict(self):
        return self._module_dict  # 등록된 모듈 딕셔너리 반환

    @property
    def children(self):
        return self._children  # 하위 레지스트리 반환

    def get(self, key):
        """레지스트리에 등록된 객체를 가져오는 함수.

        Args:
            key (str): 문자열 형식의 클래스 이름.

        Returns:
            class: 해당 클래스.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # 현재 레지스트리에서 검색
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # 하위 레지스트리에서 검색
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # 최상위 레지스트리에서 검색
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)  # build 함수 호출

    def _add_children(self, registry):
        """레지스트리에 하위 레지스트리를 추가하는 함수.

        지정된 레지스트리를 범위에 따라 하위로 추가합니다.
        부모 레지스트리에서 하위 레지스트리의 객체를 생성할 수 있게 합니다.

        예제:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"{self.name} 레지스트리에 이미 범위 {registry.scope}가 있습니다"
        self.children[registry.scope] = registry  # 자식 레지스트리로 추가

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError("module은 클래스여야 합니다, " f"그러나 {type(module_class)}를 받았습니다.")

        if module_name is None:
            module_name = module_class.__name__  # 이름이 없으면 클래스 이름을 사용
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name}는 {self.name}에 이미 등록되어 있습니다")
            self._module_dict[name] = module_class  # 모듈을 등록

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            "기존의 register_module(module, force=False) API는 더 이상 지원되지 않으며, "
            "새 API register_module(name=None, force=False, module=None)을 사용하십시오."
        )
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """모듈을 등록하는 함수.

        `self._module_dict`에 레코드를 추가하며, 키는 클래스 이름 또는 지정된 이름이고 값은 클래스 자체입니다.
        데코레이터나 일반 함수로 사용할 수 있습니다.

        예제:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): 등록할 모듈 이름. 지정하지 않으면 클래스 이름을 사용.
            force (bool, optional): 동일 이름 클래스가 있을 경우 덮어쓸지 여부. 기본값: False.
            module (type): 등록할 모듈 클래스.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force는 불리언이어야 합니다, 그러나 {type(force)}를 받았습니다")
        # 기존 API 호환성을 유지하기 위한 코드
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name은 None이거나, str의 인스턴스 또는 str 시퀀스여야 합니다, "
                f"그러나 {type(name)}를 받았습니다"
            )

        # 일반 메서드로 사용: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # 데코레이터로 사용: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register
