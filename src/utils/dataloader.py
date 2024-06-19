import copy
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input, resize_image


class FacenetDataset(Dataset):
    def __init__(self, input_shape, lines, num_classes, random, loss_type="triplet_loss"):
        self.input_shape = input_shape
        self.lines = lines
        self.length = len(lines)
        self.num_classes = num_classes
        self.random = random
        # 不同的loss对应不同的数据堆叠方式
        assert loss_type in ["triplet_loss", "arcface", "contrastive_loss"]
        self.loss_type = loss_type

        #   路径和标签
        self.paths = []
        self.labels = []
        self.label2indices = {}

        self.load_dataset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        不同的loss_type对应不同的数据堆叠方式，
        triplet_loss：3张人脸为一组；arcface：单张人脸；contrastive_loss：单张人脸
        """
        if self.loss_type == "triplet_loss":
            #   创建全为零的矩阵
            images = np.zeros((3, 3, self.input_shape[0], self.input_shape[1]))
            labels = np.zeros((3))

            #   先获得两张同一个人的人脸
            #   用来作为anchor和positive
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]
            while len(selected_path) < 2:
                c = random.randint(0, self.num_classes - 1)
                selected_path = self.paths[self.labels[:] == c]

            #   随机选择两张
            image_indexes = np.random.choice(range(0, len(selected_path)), 2)
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))
            #   翻转图像
            if self.rand() < .5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            image = preprocess_input(np.array(image, dtype='float32'))
            image = np.transpose(image, [2, 0, 1])
            images[0, :, :, :] = image
            labels[0] = c

            image = cvtColor(Image.open(selected_path[image_indexes[1]]))
            if self.rand() < .5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            image = preprocess_input(np.array(image, dtype='float32'))
            image = np.transpose(image, [2, 0, 1])
            images[1, :, :, :] = image
            labels[1] = c

            #   取出另外一个人的人脸
            different_c = list(range(self.num_classes))
            different_c.pop(c)
            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = different_c[different_c_index[0]]
            selected_path = self.paths[self.labels == current_c]
            while len(selected_path) < 1:
                different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
                current_c = different_c[different_c_index[0]]
                selected_path = self.paths[self.labels == current_c]

            image_indexes = np.random.choice(range(0, len(selected_path)), 1)
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))

            if self.rand() < .5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
            image = preprocess_input(np.array(image, dtype='float32'))
            image = np.transpose(image, [2, 0, 1])
            images[2, :, :, :] = image
            labels[2] = current_c

            return images, labels

        elif self.loss_type in ["arcface", "contrastive_loss"]:
            image_path = self.paths[index]
            label = self.labels[index]
            # 读取图像
            image = cvtColor(Image.open(image_path))
            # 随机翻转
            if self.rand() < 0.5 and self.random:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)

            mode = self.loss_type if self.loss_type == "arcface" else None
            image = preprocess_input(np.array(image, dtype='float32'), mode=mode)
            image = np.transpose(image, [2, 0, 1])
            return image, label

        else:
            raise NotImplementedError

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[0]))
        try:
            self.paths = np.array(self.paths, dtype=np.object)
        except:
            self.paths = np.array(self.paths, dtype=np.object_)

        if self.loss_type == "contrastive_loss":
            for idx, label in enumerate(self.labels):
                if label not in self.label2indices:
                    self.label2indices[label] = [idx]
                else:
                    self.label2indices[label].append(idx)
        
        self.labels = np.array(self.labels)


# DataLoader中collate_fn使用
def dataset_collate_triplet_loss(batch):
    """
    使用triplet loss训练时的collate_fn，使用arcface和contrastive loss不需要
    """
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)

    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).long()
    return images, labels


class UniqueLabelBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last):
        """
        定义batch_sampler，使得每一个batch中每个label都不一样，适用于使用contrastive loss训练
        """
        # 数据集
        self.dataset = dataset
        # batch_size
        self.batch_size = batch_size
        # 是否扔掉剩余样本
        self.drop_last = drop_last

    def __iter__(self):
        """
        每次从剩余的标签中不放回抽样，在一个batch中，每个标签对应的图片抽两张，互为正样本
        """
        batch = []
        remaining_l2i = copy.deepcopy(self.dataset.label2indices)

        while remaining_l2i:

            if len(batch) < self.batch_size:
                # 选择labels
                ks = list(remaining_l2i.keys())
                labels = np.random.choice(ks, min(len(ks), self.batch_size), replace=False)
                # 每个label选择两个idx
                for label in labels:
                    indices = remaining_l2i[label]
                    n = len(indices)
                    if n in [1, 2]:
                        if n == 1:
                            batch += remaining_l2i[label] * 2
                        elif n == 2:
                            batch += remaining_l2i[label]
                        # 更新
                        remaining_l2i.pop(label)
                    else:
                        if n <= 0:
                            raise Exception("error")
                        idx12 = np.random.choice(indices, 2, replace=False)
                        batch += idx12.tolist()
            else:
                # 生成一个batch
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch


