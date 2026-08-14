"""
evaluator: 评估模块。

提供评估器抽象接口和具体实现。
"""
from evaluator.base import BaseEvaluator
from evaluator.downstream import DownstreamEvaluator

__all__ = ["BaseEvaluator", "DownstreamEvaluator"]
