from predict.predict import predict
from predict.predict_datasets import PredictDatasets
from predict.predict_onnxruntime import OnnxInference

if __name__ == '__main__':
    # # 1）两两匹配，看是否是同一个人
    # predict()
    # 2）对一个数据集（或其中一部分）以batch的形式做并行预测
    predict_datasets = PredictDatasets(datasets_path="/home/ubuntu/eztest_face/datasets")
    predict_datasets.predict_same_person_batch()
    predict_datasets.predict_different_persons_batch()
    predict_datasets.visualize()
    # # 3）如果模型转换成了onnx格式，可以使用onnxruntime推理
    # onnx_inference = OnnxInference("your_onnx_model.onnx", "cuda")
