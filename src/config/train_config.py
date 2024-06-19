from typing import List, Optional
from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    """
    训练配置
    """

    def __init__(self,
                 # I/O
                 save_dir: str = "logs",  # 模型存储文件夹
                 save_period: int = 2,  # 模型保存周期
                 backbone: str = "mobilenet",  # facenet的backbone
                 model_path: str = "model_data/facenet_mobilenet.pth",  # 预训练模型路径
                 input_shape: Optional[List[int]] = None,  # 输入尺寸

                 batch_size: int = 96,    # 576,  # 训练集batch_size
                 val_batch_size: int = 48,  # 验证集batch_size
                 Init_Epoch: int = 0,  # 初始epoch
                 Epoch: int = 100,  # 总的epoch

                 annotation_path: str = "cls_train.txt",  # 训练集标注数据
                 val_annotation_path: str = "cls_val.txt",  # 验证集标注数据

                 # optimizer
                 optimizer_type: str = "adam",  # 优化器类型
                 momentum: float = 0.9,  # 动量
                 weight_decay: float = 0.0,  # 权重衰减因子

                 # lr -> 需要计算一下
                 nbs: Optional[int] = 64,  # 我也不知道是个啥
                 Init_lr: float = 1e-3,  # 最大学习率
                 Min_lr: Optional[float] = None,  # 最小学习率，默认会计算一个值
                 lr_decay_type: str = "cos",      # 学习率下降类型

                 loss_type: str = "arcface",  # loss类型   ["triplet_loss", "arcface", "contrastive_loss"]

                 # system
                 device: str = "cuda",  # 设备
                 num_workers: int = 4,  # workers个数，可加速
                 fp16: bool = True,  # 是否采用混合精度训练
                 seed: int = 11,  # 随机种子
                 ):
        # I/O
        self.save_dir = save_dir
        self.save_period = save_period
        self.backbone = backbone
        self.model_path = model_path
        self.input_shape = [160, 160, 3] if input_shape is None else input_shape

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.Init_Epoch = Init_Epoch
        self.Epoch = Epoch

        self.annotation_path = annotation_path
        self.val_annotation_path = val_annotation_path

        # optimizer
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.weight_decay = weight_decay

        # lr
        self.nbs = nbs
        self.Init_lr = Init_lr
        self.Min_lr = Min_lr
        self.lr_decay_type = lr_decay_type

        self.loss_type = loss_type

        # system
        self.device = device
        self.num_workers = num_workers
        self.fp16 = fp16
        self.seed = seed

        self._init()

    def _init(self):
        """
        初始化，有部分参数需要计算
        """
        if self.Min_lr is None:
            self.Min_lr = self.Init_lr * 0.01

        lr_limit_max = 1e-3 if self.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if self.optimizer_type == 'adam' else 5e-4
        # 实际计算的lr取值，如果觉得不合适，也可以自己直接手动修改
        self.Init_lr_fit = min(max(self.batch_size / self.nbs * self.Init_lr, lr_limit_min), lr_limit_max)
        self.Min_lr_fit = min(max(self.batch_size / self.nbs * self.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        # self.Min_lr_fit = 5e-6

        if self.loss_type == "triplet_loss" and self.batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3 if triplet_loss used.")

        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
