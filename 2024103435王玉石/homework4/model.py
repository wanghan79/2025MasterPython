# coding="gbk"

from keras.layers import Input, Conv2D, MaxPooling2D, Layer
from keras.layers import Concatenate, UpSampling2D, BatchNormalization
from keras.models import Model


class DownBlock(Layer):

    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(width, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(width, 3, activation='relu', padding='same')
        self.maxp = MaxPooling2D()
        self.bn = BatchNormalization()

    def call(self, inputs):
        x = inputs
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.maxp(x), x


class UpBlock(Layer):

    def __init__(self, width, **kwargs):
        super().__init__(**kwargs)
        self.up_sample = UpSampling2D(interpolation="nearest")
        self.conv1 = Conv2D(width, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(width, 3, activation='relu', padding='same')
        self.conc = Concatenate()
        self.bn = BatchNormalization()

    def call(self, inputs):
        x, skip = inputs
        x = self.up_sample(x)
        x = self.conc([x, skip])
        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class UNet(Model):

    def __init__(self, output_channel, **kwargs):
        super().__init__(**kwargs)
        self.skips = []     
        self.down_blocks = [
            DownBlock(32),
            DownBlock(64),
            DownBlock(128)]
        self.conv_block = [
            Conv2D(256, 3, activation='relu', padding='same'),
            Conv2D(128, 3, activation='relu', padding='same')
            ]
        self.up_blocks = [
            UpBlock(128),
            UpBlock(64),
            UpBlock(32)
            ]
        self.out = Conv2D(output_channel, 1, activation='sigmoid')

    def call(self, inputs):
        x = inputs
        for block in self.down_blocks:
            x, skip = block(x)
            self.skips.append(skip)

        for block in self.conv_block:
            x = block(x)

        for block in self.up_blocks:
            skip = self.skips.pop()
            x = block([x, skip])
        return self.out(x)

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))






