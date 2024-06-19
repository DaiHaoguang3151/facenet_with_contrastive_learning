import os
import json
import math
import random

from PIL import Image
import matplotlib.pyplot as plt

from .facenet import Facenet


class PredictDatasets:
    """
    对于一个数据集，可以集中进行预测，打batch
    """

    def __init__(self,
                 model: Facenet = Facenet(),
                 datasets_path: str = "datasets",
                 batch_size: int = 64,
                 l1_list_file_path: str = "./l1_list.json",
                 l1_paired_image_map_file_path: str = "./l1_paired_image_map.json",
                 diff_l1_list_file_path: str = "./diff_l1_list.json",
                 diff_l1_paired_image_map_file_path: str = "./diff_l1_paired_image_map.json",
                 ):
        # 预测使用的模型
        self.model = model
        # 需要预测的dataset所在目录
        self.datasets_path = datasets_path
        # batch_size
        self.batch_size = batch_size
        # l1距离列表
        self.l1_list_file_path = l1_list_file_path
        # 哪两张图片对应的l1距离
        self.l1_paired_image_map_file_path = l1_paired_image_map_file_path
        # 【不同人】之间的l1距离列表
        self.diff_l1_list_file_path = diff_l1_list_file_path
        # 哪两张图片对应的l1距离
        self.diff_l1_paired_image_map_file_path = diff_l1_paired_image_map_file_path
        # 报名照：登录照
        self.signup_login_map = {}

        self._get_signup_login_map()

    def _get_signup_login_map(self):
        """
        获取同一个人的报名和登录照片的映射关系
        """
        if self.signup_login_map:
            self.signup_login_map = {}
        types_name = os.listdir(self.datasets_path)
        types_name = sorted(types_name)

        for cls_id, type_name in enumerate(types_name):
            photos_path = os.path.join(self.datasets_path, type_name)
            if not os.path.isdir(photos_path):
                continue
            photos_name = sorted(os.listdir(photos_path))

            if len(photos_name) < 2:
                continue
            # if not photos_name[0].endswith("0.jpg"):
            #     print("something wrong")
            #     continue
            self.signup_login_map.update(
                {os.path.join(photos_path, photos_name[i]): os.path.join(photos_path, photos_name[0]) for i in
                 range(1, len(photos_name))})

    def predict_same_person_batch(self):
        """
        同一个人的几张照片进行距离计算，假设第一张照片是【报名照】，后续的照片都是【登录照】
        """

        n = len(self.signup_login_map)
        keys = list(self.signup_login_map.keys())
        num = math.ceil(n / self.batch_size)

        l1_list = []
        l1_paired_image_map = {}

        for i in range(0, num):
            # 取一个batch预测
            print(f"{i + 1} / {num}")
            lower = i * self.batch_size
            upper = min((i + 1) * self.batch_size, n)
            keys_batch = keys[lower: upper]

            # 读取image
            target_images = []
            target_paths = []
            for key in keys_batch:
                val = self.signup_login_map[key]
                try:
                    image1 = Image.open(key)
                    image2 = Image.open(val)

                except:
                    image1 = None
                    image2 = None
                    continue
                target_images += [image1, image2]
                target_paths += [key, val]
            # 进入模型
            l1_batch_list = self.model.detect_image_batch(target_images)
            l1_list += l1_batch_list

            if len(l1_batch_list) == len(target_paths) / 2:
                for j in range(0, len(l1_batch_list)):
                    k = str((target_paths[2 * j], target_paths[2 * j + 1]))
                    v = l1_batch_list[j]
                    l1_paired_image_map.update({k: v})

        with open(self.l1_list_file_path, "w", encoding="utf-8") as f:
            json.dump(l1_list, f)

        with open(self.l1_paired_image_map_file_path, "w", encoding="utf-8") as f:
            json.dump(l1_paired_image_map, f)
        print(len(l1_paired_image_map))
        return l1_list

    def predict_different_persons_batch(self):
        """
        不同人之间照片距离计算
        """
        n = len(self.signup_login_map)
        keys = list(self.signup_login_map.keys())
        num = math.ceil(n / self.batch_size)

        diff_l1_list = []
        diff_l1_paired_image_map = {}

        random.seed(11)

        for i in range(0, num):
            # 取一个batch预测
            print(f"{i + 1} / {num}")
            lower = i * self.batch_size
            upper = min((i + 1) * self.batch_size, n)
            keys_batch = keys[lower: upper]

            # 读取image
            target_images = []
            target_paths = []

            if len(keys_batch) < 2:
                continue
            for j, key in enumerate(keys_batch):
                # 随便sample两个数据
                tmp_indices = []
                while len(tmp_indices) < 2:
                    idx = random.randint(0, len(keys_batch) - 1)
                    if idx == j:
                        continue
                    tmp_indices.append(idx)
                assert len(tmp_indices) == 2
                candidate1 = keys_batch[tmp_indices[0]]
                candidate2 = self.signup_login_map[keys_batch[tmp_indices[1]]]

                try:
                    image1 = Image.open(key)
                    image2 = Image.open(candidate1)
                    image3 = Image.open(candidate2)

                except:
                    image1 = None
                    image2 = None
                    image3 = None
                    continue
                target_images += [image1, image2, image1, image3]
                target_paths += [key, candidate1, key, candidate2]

            diff_l1_batch_list = self.model.detect_image_batch(target_images)
            diff_l1_list += diff_l1_batch_list

            if len(diff_l1_batch_list) == len(target_paths) / 2:
                for j in range(0, len(diff_l1_batch_list)):
                    k = str((target_paths[2 * j], target_paths[2 * j + 1]))
                    v = diff_l1_batch_list[j]
                    diff_l1_paired_image_map.update({k: v})

        with open(self.diff_l1_list_file_path, "w", encoding="utf-8") as f:
            json.dump(diff_l1_list, f)

        with open(self.diff_l1_paired_image_map_file_path, "w", encoding="utf-8") as f:
            json.dump(diff_l1_paired_image_map, f)
        print(len(diff_l1_paired_image_map))
        return diff_l1_list

    def visualize(self):
        """
        可视化统计结果
        """
        if os.path.exists(self.l1_list_file_path):
            with open(self.l1_list_file_path, "r", encoding="utf-8") as f1:
                data_same_person = json.load(f1)
            bin_counts1, bin_edges1, _ = plt.hist(data_same_person, bins=20, label="Same person", alpha=0.5)
        if os.path.exists(self.diff_l1_list_file_path):
            with open(self.diff_l1_list_file_path, "r", encoding="utf-8") as f2:
                data_diff_persons = json.load(f2)
            bin_counts2, bin_edges2, _ = plt.hist(data_diff_persons, bins=20, label="Same person", alpha=0.5)

        # 设置柱状图标题和轴标签
        plt.title('Histogram of Embedding Distance')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        plt.legend()

        # 显示柱状图
        plt.show()


if __name__ == '__main__':
    predict_datasets = PredictDatasets()
    predict_datasets.predict_same_person_batch()
    predict_datasets.predict_different_persons_batch()
    predict_datasets.visualize()
