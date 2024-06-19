import os


def datasets_assembly(
        datasets_path_list,  # 数据集路径列表
        sample_file_path_list,  # 有一些数据集是经过采样的，相应的输入是一个txt文件
        list_file_path  # 输出文件路径
):
    """
    多个不同来源的数据集进行组合，构成新的数据集用于训练
    """
    datasets_path_types_name_map = {}  # 这边存一下，主要用于检查是否存在如下情况：不同的dataset，但是里面有同名的文件夹
    types_name_total = []

    for datasets_path in datasets_path_list:
        types_name = os.listdir(datasets_path)
        types_name = sorted(types_name)
        datasets_path_types_name_map.update({datasets_path: types_name})
        types_name_total += types_name

    # 检查是否存在同名文件夹
    n = len(types_name_total)
    to_remove_types_name = []
    types_name_total_set = set(types_name_total)
    if len(types_name_total_set) != n:
        from collections import Counter
        element_count = Counter(types_name_total)
        to_remove_types_name = [k for k, v in element_count.items() if v > 1]
        print(f"重名个数为{len(to_remove_types_name)}")
        # raise Exception("重名了，检查一下")

    # 刨除重名数据
    for datasets_path, types_name in datasets_path_types_name_map.items():
        for name in to_remove_types_name:
            if name in types_name:
                # 删除
                idx = types_name.index(name)
                types_name.pop(idx)
        datasets_path_types_name_map[datasets_path] = types_name

    cls_id = -1
    with open(list_file_path, "w", encoding="utf-8") as f:
        for datasets_path in datasets_path_list:
            types_name = datasets_path_types_name_map[datasets_path]
            for type_name in types_name:
                cls_id += 1

                photos_path = os.path.join(datasets_path, type_name)
                if not os.path.isdir(photos_path):
                    continue
                photos_name = os.listdir(photos_path)
                for photo_name in photos_name:
                    f.write(str(cls_id) + ";" + '%s' % (
                        os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
                    f.write('\n')
        print("cls===> ", cls_id)

        # 追加采样数据集（默认不会重名）
        for sample_file_path in sample_file_path_list:
            with open(sample_file_path, "r", encoding="utf-8") as f2:
                lines = f2.readlines()
                num_ = int(lines[-1].split(";")[0])
                # 修改其中的cls_id，写道文件中
                for line in lines:
                    origin_cls_id, other_info = line.split(";")
                    new_cls_id = int(origin_cls_id) + cls_id
                    new_line = str(new_cls_id) + ";" + other_info
                    f.write(new_line)
                cls_id += (num_ + 1)
    print("total==> ", cls_id)
    return


if __name__ == "__main__":
    datasets_assembly(["/home/ubuntu/eztest_face/datasets",
                       "/home/ubuntu/eztest_face/datasets_2"],
                      sample_file_path_list=["datasets_MPIIFaceGaze_cls_train.txt", "CASIA-WebFace_cls_train.txt"],
                      list_file_path="assembly_all_datasets_cls_train.txt")
