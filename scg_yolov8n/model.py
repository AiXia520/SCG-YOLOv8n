from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG


# Import our custom modules so YAML can resolve them
from .modules import CSPD as CSPD
from .modules import SCARAFE as SCARAFE
from .modules import GhostShuffle as GhostShuffle


import os

def build_scg_yolov9n(model_yaml: str, nc: int = 1):
    """Build a YOLO model from YAML with our custom layers registered."""
    # YOLO class automatically registers globals in its namespace if imported
    cfg = get_cfg(DEFAULT_CFG)
    cfg.model = model_yaml
    model = YOLO(cfg)
    model.model.nc = nc
    return model
