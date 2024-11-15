"""
Hook Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry


# 훅을 관리하는 레지스트리
HOOKS = Registry("hooks")


# 설정을 기반으로 훅을 빌드하는 함수
def build_hooks(cfg):
    hooks = []  # 훅을 저장할 리스트
    for hook_cfg in cfg:  # 설정된 각 훅 구성 요소에 대해
        hooks.append(HOOKS.build(hook_cfg))  # 해당 훅을 레지스트리에서 빌드하여 추가
    return hooks  # 빌드된 훅 목록을 반환
