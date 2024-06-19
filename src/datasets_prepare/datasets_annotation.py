import os


def datasets_annotation(datasets_path, out_file_path):
    """
    生成用于训练和验证的txt文件
    """
    types_name = os.listdir(datasets_path)
    types_name = sorted(types_name)

    list_file = open(out_file_path, 'w')
    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            list_file.write(
                str(cls_id) + ";" + '%s' % (os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
            list_file.write('\n')
    list_file.close()


if __name__ == '__main__':
    datasets_annotation("/home/ubuntu/eztest_face/datasets", "cls_train.txt")
