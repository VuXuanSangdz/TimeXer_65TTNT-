"""Model registry for RQ1 comparison experiments."""

from .DLinear import DLinear
from .PatchTST import PatchTST
from .iTransformer import iTransformer
from .TiDE import TiDE
from .TimeXer import TimeXer

BASELINE_MODELS = {
    "TimeXer": TimeXer,
    "iTransformer": iTransformer,
    "PatchTST": PatchTST,
    "TiDE": TiDE,
    "DLinear": DLinear,
}
