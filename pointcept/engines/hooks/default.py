"""
Default Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


class HookBase:
    """
    TrainerBase 클래스에 등록할 수 있는 훅의 기본 클래스입니다.
    """

    trainer = None  # Trainer 객체에 대한 약한 참조입니다.

    def before_train(self):
        pass  # 훈련 시작 전 호출되는 메서드입니다.

    def before_epoch(self):
        pass  # 각 에포크 시작 전 호출되는 메서드입니다.

    def before_step(self):
        pass  # 각 스텝 시작 전 호출되는 메서드입니다.

    def after_step(self):
        pass  # 각 스텝 종료 후 호출되는 메서드입니다.

    def after_epoch(self):
        pass  # 각 에포크 종료 후 호출되는 메서드입니다.

    def after_train(self):
        pass  # 훈련 종료 후 호출되는 메서드입니다.
