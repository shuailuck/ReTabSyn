"""
scenarios/__init__.py: 导出所有 Scenario 类与 Factory 统一调用入口
"""

from scenario import BaseScenario
from small_data import SmallDataScenario
from imbalanced import ImbalancedDataScenario
from distribution_shift import DistributionShiftScenario


class ScenarioFactory:
    """场景工厂类"""

    _scenarios = {
        "small": SmallDataScenario,
        "imbalanced": ImbalancedDataScenario,
        "shift": DistributionShiftScenario,
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
    "ScenarioFactory",
]