import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config.predict_config import PredictConfig
from nets.facenet import Facenet as facenet
from utils.utils import preprocess_input, resize_image, show_config

predict_config = PredictConfig(model_path="model_data/facenet_mobilenet.pth")


class Facenet(object):
    _defaults = {
        # 模型路径
        "model_path": predict_config.model_path,
        # 输入尺寸
        "input_shape": predict_config.input_shape,
        # 主干特征提取网络
        "backbone": predict_config.backbone,
        # 是否进行不失真的resize
        "letterbox_image": predict_config.letterbox_image,
        # 设备
        "cuda": predict_config.device.startswith("cuda"),
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        """
        初始化Facenet
        """
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
        """
        载入模型与权值, onnx=True表示需要转onnx
        """
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = True
                self.net = self.net.cuda()

    def _preprocess_image(self, image):
        """
        输入模型前的图像预处理
        """
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
        image = np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1))
        return image

    def detect_image(self, image_1, image_2):
        """
        检测两张图片中是否是同一个人
        """
        with torch.no_grad():
            photo_1 = torch.from_numpy(np.expand_dims(self._preprocess_image(image_1), 0))
            photo_2 = torch.from_numpy(np.expand_dims(self._preprocess_image(image_2), 0))

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # 图片传入网络进行预测
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()

            # 计算二者之间的距离
            l1 = np.linalg.norm(output1 - output2, axis=1)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va='bottom', fontsize=11)
        plt.show()
        return l1

    def detect_image_batch(self, images):
        """
        批量预测：对应场景是有一个数据集，数据量比较大，就需要合成batch批量检测，这样会加速很多
        两张图片为一个对比单元，多个单元堆叠成一个batch
        """
        with torch.no_grad():
            image_array = []
            for image in images:
                image = self._preprocess_image(image).tolist()
                image_array.append(image)
            image_array = np.asarray(image_array, dtype=np.float32)
            photos = torch.from_numpy(image_array)

            if self.cuda:
                photos.cuda()
            output = self.net(photos).cpu().numpy()

            # 计算两两之间的距离
            num = output.shape[0]
            if num % 2 == 1:
                raise Exception("没有两两成对")
            l1_list = []
            for i in range(0, num, 2):
                l1 = np.linalg.norm(output[i] - output[i + 1]).tolist()
                l1_list.append(l1)
        return l1_list

    def convert_to_onnx(self, simplify, model_path):
        """
        转成onnx, model_path保存的模型的路径，是必要参数
        """
        import onnx
        self.generate(onnx=True)  # 这边是加载模型

        im = torch.zeros(1, 3, *self.input_shape[:2]).to("cpu")
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # 导出模型
        print(f"Starting export with onnx {onnx.__version__}.")
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # 检查模型
        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        # simplify
        if simplify:
            import onnxsim
            print(f"Simplifying with onnx-simplifier {onnxsim.__version__}.")
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None
            )
            assert check, "assert check failed"
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
