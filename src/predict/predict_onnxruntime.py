from typing import Union, List, Tuple

import numpy as np
import onnxruntime


# 这边理论上能实现predict(_datasets)功能，但不想重复写了
class OnnxInference:
    """
    传入onnx模型，即可推理获得embedding
    """
    def __init__(self, onnx_model_path: str, device_name: str):
        # onnx模型路径
        self.onnx_model_path = onnx_model_path
        # 设备
        self.device_name = device_name
        # providers
        self.providers = ["CPUExecutionProvider"] if device_name == "cpu" \
            else ['CUDAExecutionProvider', 'CPUExecutionProvider']   # 应该还有一个triton的Provider
        self.session = onnxruntime.InferenceSession(self.onnx_model_path, providers=self.providers)
        # 推理时需要获取输入名称
        self.input_name = self.session.get_inputs()[0].name
        # 输出名称
        self.output_names = [output.name for output in self.session.get_outputs()]

    def infer(self, imgs_array: np.ndarray):
        """
        推理：可以输入多张图片
        """
        return self.session.run(self.output_names, input_feed={self.input_name: imgs_array})

    def get_fps(self, model_shape: Union[List, Tuple]):
        """
        获取fps以查看推理速度，每次输入单张图片
        model_shape：模型shape，不同框架训练出来的model_shape是不一样的，比如channel维度放在前面还是后面
        """
        import time
        # img = np.ones((1, 3, 640, 640))    # pytorch + yolo
        # img = np.ones((1, 160, 160, 3))     # keras + tf + facenet
        img = np.ones(tuple(model_shape))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            res = self.infer(img)
            print(res[0])

        t0 = time.perf_counter()
        for _ in range(100):
            _ = self.infer(img)
        print(100 / (time.perf_counter() - t0), 'FPS')