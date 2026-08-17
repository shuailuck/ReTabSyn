"""
scenarios/__init__.py: 导出所有 Scenario 类与 Factory 统一调用入口
"""

from scenario.scenario import BaseScenario
from scenario.small_data import SmallDataScenario
from scenario.imbalanced import ImbalancedDataScenario
from scenario.distribution_shift import DistributionShiftScenario
from scenario.noisy_label import NoisyLabelScenario, EvolveNoisyLabelScenario


class ScenarioFactory:
    """场景工厂类。"""

    _scenarios = {
        "small": SmallDataScenario,
        "imbalanced": ImbalancedDataScenario,
        "shift": DistributionShiftScenario,
        "noisy_label": NoisyLabelScenario,
    }

    # 场景变体: (scenario_name, augment_method) -> 场景类
    _variants = {
        ("noisy_label", "evolve"): EvolveNoisyLabelScenario,
    }

    @classmethod
    def create(cls, scenario_name: str, **kwargs) -> BaseScenario:
        name = scenario_name.lower()
        if name not in cls._scenarios:
            raise ValueError(f"未知场景 '{scenario_name}'，可选场景: {list(cls._scenarios.keys())}")

        # 通过 augment_method 选择场景变体（如 evolve 对应 EvolveNoisyLabelScenario）
        augment_method = kwargs.pop("augment_method", None)
        scenario_cls = cls._variants.get((name, augment_method), cls._scenarios[name])
        return scenario_cls(**kwargs)


__all__ = [
    "BaseScenario",
    "SmallDataScenario",
    "ImbalancedDataScenario",
    "DistributionShiftScenario",
    "NoisyLabelScenario",
    "EvolveNoisyLabelScenario",
    "ScenarioFactory",
]
