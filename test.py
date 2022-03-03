from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.backend import sum, binary_crossentropy, mean


def get_mod(rate: float, filters: int = 32):
    x = Input(shape=(32, 32, 3))
    y = Conv2D(filters=filters, kernel_size=(1, 1), padding="same")(x)
    y = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(rate)(y)
    return Model(inputs=x, outputs=y)


mod = get_mod(0.2)
plot_model(mod, to_file='mode.png', show_shapes=True)


def soft_dice_loss_and_binary(y_true, y_pred):
    addition = sum(y_true * y_pred, axis=[1, 2, 3])
    union = sum(y_true, axis=[1 ,2 ,3]) + sum(y_pred, axis=[1, 2, 3])
    return (1 - mean((2. * addition + 1.) / (union + 1.)) +
            binary_crossentropy(y_true, y_pred)) / 2

