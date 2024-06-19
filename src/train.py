import os
from functools import partial
from typing import Optional, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loss.contrastive_loss import contrastive_loss
from loss.triplet_loss import triplet_loss
from nets.facenet import Facenet
from nets.arcface import ArcMarginProduct
from nets.facenet_training import get_lr_scheduler, set_optimizer_lr
from utils.dataloader import FacenetDataset, dataset_collate_triplet_loss, UniqueLabelBatchSampler
from utils.utils import get_annotated_data, seed_everything, get_lr

from config.train_config import TrainConfig


class Trainer:
    def __init__(self,
                 train_config: TrainConfig
                 ):
        # 训练配置
        self.train_config: TrainConfig = train_config

        # 训练集类别数
        self.num_classes: int = 0
        # 验证集类别数
        self.val_num_classes: int = 0
        # 训练集数据
        self.lines: List[str] = []
        # 验证集数据
        self.val_lines: List[str] = []

        # facenet模型
        self.model: Facenet = None
        # 使用的margin
        self.margin: Optional[ArcMarginProduct] = None
        # facenet的mode，决定了facenet的分类头是否使用
        self.facenet_mode = None

        # 优化器
        self.optimizer: optim.Optimizer = None
        # 学习率公式
        self.lr_scheduler_func: partial = None
        # 训练集生成器
        self.gen = None
        # 验证集生成器
        self.val_gen = None

        self._init()

    def _init(self):
        """
        初始化
        """
        seed_everything(self.train_config.seed)
        self.num_classes, self.lines = get_annotated_data(self.train_config.annotation_path)
        self.val_num_classes, self.val_lines = get_annotated_data(self.train_config.val_annotation_path)
        # 必要检查
        self._check()
        # 模型初始化
        self._init_model()
        # 设置优化器
        self._set_optimizer()
        # 获取学习率下降公式
        self._get_lr_scheduler()
        # 获取数据集
        self._get_datasets()

    def _check(self):
        """
        检查输入参数是否合规
        """
        if self.train_config.loss_type == "triplet_loss" and self.train_config.batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3 if triplet_loss used.")

    def _init_model(self):
        """
        初始化模型
        """
        # 当采用triplet loss的时候，Facenet中的分类头也是需要的
        self.facenet_mode = "predict" if self.train_config.loss_type != "triplet_loss" else "train"

        # facenet
        self.model = Facenet(backbone=self.train_config.backbone,
                             num_classes=self.num_classes,
                             mode=self.facenet_mode)
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(self.train_config.model_path, map_location=self.train_config.device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.train_config.device)

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

        # arcface
        if self.train_config.loss_type == "arcface":
            self.margin = ArcMarginProduct(128, self.num_classes, s=32)
            self.margin.to(self.train_config.device)

    def _set_optimizer(self):
        """
        设置优化器
        """
        params = [{"params": self.model.parameters()}]
        if self.train_config.loss_type == "arcface":
            params.append({"params": self.margin.parameters()})
        optimizers = {
            "adam": optim.Adam(params, self.train_config.Init_lr_fit,
                               betas=(self.train_config.momentum, 0.999),
                               weight_decay=self.train_config.weight_decay),
            "sgd": optim.SGD(params, self.train_config.Init_lr_fit,
                             momentum=self.train_config.momentum,
                             nesterov=True,
                             weight_decay=self.train_config.weight_decay)
        }
        self.optimizer = optimizers[self.train_config.optimizer_type]

    def _get_lr_scheduler(self):
        """
        获取lr公式
        """
        self.lr_scheduler_func = get_lr_scheduler(self.train_config.lr_decay_type,
                                                  self.train_config.Init_lr_fit,
                                                  self.train_config.Min_lr_fit,
                                                  self.train_config.Epoch)

    def _get_datasets(self):
        """
        获取训练集和测试集
        """
        np.random.seed(10101)
        np.random.shuffle(self.lines)
        np.random.shuffle(self.val_lines)
        np.random.seed(None)

        # dataset
        train_dataset = FacenetDataset(input_shape=self.train_config.input_shape,
                                       lines=self.lines,
                                       num_classes=self.num_classes,
                                       random=True,
                                       loss_type=self.train_config.loss_type)

        val_dataset = FacenetDataset(input_shape=self.train_config.input_shape,
                                     lines=self.val_lines,
                                     num_classes=self.val_num_classes,
                                     random=True,
                                     loss_type=self.train_config.loss_type)
        # dataloader
        num_workers = self.train_config.num_workers
        pin_memory = True

        shuffle = True
        batch_size = self.train_config.batch_size
        val_batch_size = self.train_config.val_batch_size
        drop_last = False
        collate_fn = None
        sampler = None
        batch_sampler = None
        val_batch_sampler = None

        if self.train_config.loss_type == "triplet_loss":
            batch_size = batch_size // 3
            val_batch_size = val_batch_size // 3
            drop_last = True
            collate_fn = dataset_collate_triplet_loss
        elif self.train_config.loss_type == "contrastive_loss":
            shuffle = False
            batch_size = 1
            val_batch_size = 1
            batch_sampler = UniqueLabelBatchSampler(train_dataset, self.train_config.batch_size, drop_last=False)
            val_batch_sampler = UniqueLabelBatchSampler(val_dataset, self.train_config.val_batch_size, drop_last=False)

        self.gen = DataLoader(train_dataset,
                              shuffle=shuffle,
                              batch_size=batch_size,
                              pin_memory=pin_memory,
                              num_workers=num_workers,
                              drop_last=drop_last,
                              collate_fn=collate_fn,
                              sampler=sampler,
                              batch_sampler=batch_sampler,
                              worker_init_fn=None  # todo
                              )

        self.val_gen = DataLoader(val_dataset,
                                  shuffle=shuffle,  # or False
                                  batch_size=val_batch_size,
                                  pin_memory=pin_memory,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  collate_fn=collate_fn,
                                  sampler=sampler,
                                  batch_sampler=val_batch_sampler,
                                  worker_init_fn=None
                                  )

    def train(self):
        """
        训练
        """
        epoch_step = len(self.lines) // self.train_config.batch_size
        val_epoch_step = len(self.val_lines) // self.train_config.val_batch_size

        if self.train_config.fp16:
            from torch.cuda.amp import GradScaler as GradScaler
            scaler = GradScaler()
        else:
            scaler = None

        for epoch in range(self.train_config.Init_Epoch, self.train_config.Epoch):
            # 设置学习率
            set_optimizer_lr(self.optimizer, self.lr_scheduler_func, epoch)
            self.model.train()
            total_loss = 0

            print("Start Train")
            pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{self.train_config.Epoch}', postfix=dict,
                        mininterval=0.3)

            for iteration, batch in enumerate(self.gen):
                if iteration >= epoch_step:
                    break
                images, labels = batch

                with torch.no_grad():
                    images = images.to(self.train_config.device)
                    labels = labels.to(self.train_config.device)
                self.optimizer.zero_grad()

                if not self.train_config.fp16:
                    _loss = self._forward(images, labels, train=True)
                    _loss.backward()
                    self.optimizer.step()
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        _loss = self._forward(images, labels, train=True)
                    scaler.scale(_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                total_loss += _loss.item()

                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
            pbar.close()
            print("Finish Train")

            # 验证
            self.model.eval()
            val_total_loss = 0
            val_pbar = tqdm(total=val_epoch_step, desc=f"Epoch {epoch + 1}/{self.train_config.Epoch}", postfix=dict,
                            mininterval=0.3)

            for val_iteration, val_batch in enumerate(self.val_gen):
                if val_iteration >= val_epoch_step:
                    break
                val_images, val_labels = val_batch

                with torch.no_grad():
                    val_images = val_images.to(self.train_config.device)
                    val_labels = val_labels.to(self.train_config.device)
                self.optimizer.zero_grad()

                if not self.train_config.fp16:
                    _val_loss = self._forward(val_images, val_labels, train=False)
                else:
                    from torch.cuda.amp import autocast
                    with autocast():
                        _val_loss = self._forward(val_images, val_labels, train=False)
                val_total_loss += _val_loss.item()

                val_pbar.set_postfix(**{'val_total_loss': val_total_loss / (val_iteration + 1),
                                        'lr': get_lr(self.optimizer)})
                val_pbar.update(1)
            val_pbar.close()
            print("Finish Validation")

            print('Epoch:' + str(epoch + 1) + '/' + str(self.train_config.Epoch))
            if not ((epoch + 1) % self.train_config.save_period == 0 or epoch + 1 == self.train_config.Epoch):
                continue
            if not os.path.exists(self.train_config.save_dir):
                os.makedirs(self.train_config.save_dir, exist_ok=True)
            torch.save(self.model.state_dict(),
                       os.path.join(self.train_config.save_dir,
                                    'ep%03d-loss%.3f-val_loss%.3f.pth' %
                                    ((epoch + 1), total_loss / epoch_step, val_total_loss / max(1, val_epoch_step))))

    def _forward(self, images, labels, train):
        """
        根据不同的loss类型执行不同的forward逻辑
        """
        if train:
            self.model.train()
            # self.margin.train()   # 这边可以不要，因为没有dropout,batchnorm这些
        else:
            self.model.eval()
        outputs = self.model(images, self.facenet_mode)
        if self.train_config.loss_type == "triplet_loss":
            outputs1, outputs2 = outputs
            _batch_size = (self.train_config.batch_size if train else self.train_config.val_batch_size) // 3
            _triplet_loss = triplet_loss()(outputs1, _batch_size)  # todo: device
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss  # 这边可以自己调比例系数
        elif self.train_config.loss_type == "arcface":
            outputs_arc = self.margin(outputs, labels, self.train_config.fp16)
            criterion = nn.CrossEntropyLoss().to(self.train_config.device)
            _loss = criterion(outputs_arc, labels)
        else:
            _loss = contrastive_loss(outputs, device=self.train_config.device)
        return _loss


if __name__ == '__main__':
    trainer = Trainer(TrainConfig())
    trainer.train()