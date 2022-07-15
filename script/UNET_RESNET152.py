
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet152


###################################
def conv_block(input, filters):

    """ for UNET we need two blocks of Convolution layer
        input required:
            1. input: size of the image and in this case is (height,width,number of channels = 256, 256, 3)
            2. filters: this is the number of output filters in the convolution """

    x = Conv2D(filters, 5, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, filters):

    """To build a decoder block"""
    x = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, filters)
    return x

def UNET_RESNET152(n_classes, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):

    """ Input """
    inputs = Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

    """ initialize pretrain ResNet150 Model """
    resnet152 = ResNet152(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder_Layer as skip connection"""

    skip_connection_1 = resnet152.get_layer("input_1").output               # Input layer from resnet152 :256*256
    skip_connection_2 = resnet152.get_layer("conv1_relu").output            # output layer with size :128*128
    skip_connection_3 = resnet152.get_layer("conv2_block3_out").output      # output layer with size : 64*64
    skip_connection_4 = resnet152.get_layer("conv3_block8_out").output      # output layer with size : 32*32
    #print(skip_connection_1.shape, skip_connection_2.shape, skip_connection_3.shape, skip_connection_4.shape)

    """ Encoder to Decoder bridge"""

    connect_encoder_to_decoder = resnet152.get_layer("conv4_block1_out").output   # Output layer with size : 16*16

    """ Decoder_Block"""

    decoder_1 = decoder_block(connect_encoder_to_decoder, skip_connection_4, 512)                     
    decoder_2 = decoder_block(decoder_1, skip_connection_3, 256)                     
    decoder_3 = decoder_block(decoder_2, skip_connection_2, 128)                     
    decoder_4 = decoder_block(decoder_3, skip_connection_1, 64)                      

    """ Output """
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(decoder_4)

    model = Model(inputs, outputs, name="ResNet152_U-Net")
    return model

if __name__ == "__main__":
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    IMAGE_CHANNELS = 3
    n_class = 9
    model = UNET_RESNET152(n_classes, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    model.summary()