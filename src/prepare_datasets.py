from datasets_prepare.datasets_annotation import datasets_annotation

if __name__ == "__main__":
    datasets_annotation("/home/ubuntu/eztest_face/datasets", "cls_train.txt")
    datasets_annotation("/home/ubuntu/eztest_face/datasets", "cls_val.txt")