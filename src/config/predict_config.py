from typing import List, Optional
from dataclasses import dataclass

import torch


@dataclass
class PredictConfig:
    """
    训练配置
    """

    def __init__(self,
                 model_path: str,
                 input_shape: Optional[List[int]] = None,
                 backbone: str = "mobilenet",
                 letterbox_image: bool = True,
                 device: str = "cuda"
                 ):
        self.model_path = model_path
        self.input_shape = input_shape if input_shape is not None else [160, 160, 3]
        self.backbone = backbone
        self.letterbox_image = letterbox_image
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            device = "cpu"
        self.device = device
