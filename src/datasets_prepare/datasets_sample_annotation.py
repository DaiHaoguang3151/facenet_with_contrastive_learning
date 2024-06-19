import os
import random

random.seed(11)


def datasets_sample_annotation(
        datasets_path,  # 数据集所在目录
        num_person,  # 要找多少个人的人脸
        num_image_per_person,  # 每个人取多少张照片
        list_file_path=None  # 写入的文件
):
    """
    从原有完整的数据集中采样，取出一部分用来作为我自己的数据集
    该函数的主要用意是不希望某一个数据集（数据分布）太多，影响训练结果
    这边就直接返回一个文件了（简单粗暴）
    """
    types_name = os.listdir(datasets_path)
    print("length of types_name: ", len(types_name))
    if len(types_name) < num_person:
        raise Exception("数据集太小，不够sample")
    types_name_sample = sorted(random.sample(types_name, num_person))

    if list_file_path is None:
        list_file_path = os.path.split(datasets_path)[1] + "_cls_train.txt"

    with open(list_file_path, "w", encoding="utf-8") as f:
        for cls_id, type_name in enumerate(types_name_sample):
            if (cls_id + 1) % 100 == 0:
                print(f"{cls_id + 1} / {num_person}")
            photos_path = os.path.join(datasets_path, type_name)
            if not os.path.isdir(photos_path):
                continue
            photos_name = os.listdir(photos_path)

            # 采样几张照片
            if len(photos_name) < num_image_per_person:
                print("单人照片太少")
                # 少采样，可以选择至少采样两张
                if len(photos_name) < 2:
                    raise Exception("少于2张")
                photos_name_sample = sorted(random.sample(photos_name, len(photos_name)))
            else:
                photos_name_sample = sorted(random.sample(photos_name, num_image_per_person))

            for photo_name in photos_name_sample:
                f.write(
                    str(cls_id) + ";" + '%s' % (os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
                f.write('\n')


if __name__ == '__main__':
    datasets_sample_annotation(datasets_path="datasets_MPIIFaceGaze", num_person=15, num_image_per_person=5)
