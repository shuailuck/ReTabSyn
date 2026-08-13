"""
scenarios/__init__.py: 导出所有 Scenario 类与 Factory 统一调用入口
"""

from scenario.scenario import BaseScenario
from scenario.small_data import SmallDataScenario
from scenario.imbalanced import ImbalancedDataScenario
from scenario.distribution_shift import DistributionShiftScenario
from scenario.noisy_label import NoisyLabelScenario


class ScenarioFactory:
    """场景工厂类"""

    _scenarios = {
        "small": SmallDataScenario,
        "imbalanced": ImbalancedDataScenario,
        "shift": DistributionShiftScenario,
        "noisy_label": NoisyLabelScenario,
    }

    @classmethod
    def create(cls, scenario_name: str, **kwargs) -> BaseScenario:
        name = scenario_name.lower()
        if name not in cls._scenarios:
            raise ValueError(f"未知场景 '{scenario_name}'，可选场景: {list(cls._scenarios.keys())}")
        return cls._scenarios[name](**kwargs)


__all__ = [
    "BaseScenario",
    "SmallDataScenario",
    "ImbalancedDataScenario",
    "DistributionShiftScenario",
    "NoisyLabelScenario",
    "ScenarioFactory",
]