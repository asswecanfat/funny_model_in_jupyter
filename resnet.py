
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Layer, add


class ResNetMainLayer(Layer):
    def __init__(self, filters: int, strides: int = 1, pad: str = "same"):
        super(ResNetMainLayer, self).__init__()
        self._con = Conv2D(filters=filters,
                           kernel_size=(3, 3),
                           strides=strides,
                           padding=pad)
        self._bn = BatchNormalization()
        self._relu = ReLU()
        # 1*1卷积，转换原来的输入
        self._sample = Conv2D(filters=filters,
                              kernel_size=(1, 1),
                              strides=1)

    def call(self, inputs, *args, **kwargs):
        output = self._con(inputs)
        output = self._relu(output)
        output = self._bn(output)
        return add(output, self._sample(inputs))