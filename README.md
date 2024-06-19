## Facenet + triplet_loss / arcface / contrastive_loss

---

## 特点

1. 可以使用多种loss/方式进行训练，包括triplet loss、arcface和contrastive loss；

2. 可以在数据集上批量人脸比对，并可视化统计结果。



## 步骤

1. 进入src目录：`cd src` ； 

2. 准备数据集：CASIA-WebFace数据集可去[https://github.com/bubbliiiing/facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch)获取，如果想要使用自己的数据集，请使用[https://github.com/bubbliiiing/retinaface-pytorch](https://github.com/bubbliiiing/retinaface-pytorch)截取人脸并获取align过的人脸图片，`python prepare_datasets.py` 准备训练集和验证集txt文件；

3. 训练：根据训练配置`TrainConfig` 进行训练，`python train.py` ；

4. 预测：根据预测配置`PredictConfig` 进行预测，`predict_script.py` ；

5. 模型转换：可以转换成`onnx` 格式，并使用`onnxruntime` 推理。
   
   

## Reference

[https://github.com/bubbliiiing/facenet-pytorch](https://github.com/bubbliiiing/facenet-pytorch)

注：本repo模型等部分与上述参考repo基本一致，只是针对个人需求做了一些拓展。




