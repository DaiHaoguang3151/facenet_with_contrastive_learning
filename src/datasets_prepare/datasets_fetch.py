def datasets_fetch():
    """
    收集人脸数据，可以是现成的align好的数据，
    也可以是自己的数据，使用retinaFace截取人脸部分，必要时根据landmark做align
    """
    # 1）CASIA-WebFace数据集：https://github.com/bubbliiiing/facenet-pytorch
    # 2）使用retinaFace截取人脸部分：https://github.com/bubbliiiing/retinaface-pytorch
    #  -> 使用作者提供的Retinaface_resnet50.pth模型，同时将原始预测predict做一些重写，batch预测，并行加快速度
    # 比较简单，自己实现一下
    pass