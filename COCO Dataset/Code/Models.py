import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers

assert tf.test.is_gpu_available() == True

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  
def  dice_loss(y_true, y_predict):
    return (1-dice_coef(y_true, y_predict))

def bce_dice_loss(y_true, y_predict):
    '''
    custom loss
    '''
    return binary_crossentropy(y_true, y_predict) + (1-dice_coef(y_true, y_predict))

def metrics(target,prediction):
    intersection = np.logical_and(target,prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def loss_plot(history):
    plt.plot(history.history['loss'],'r',label='Training Loss')
    plt.plot(history.history['val_loss'],'b',label='Validation Loss')
    plt.legend()
    plt.show()

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    '''returns a block of two 3x3 convolutions, each  followed by a rectified linear unit (ReLU)'''
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def build_unet(input_img,n_filters, dropout, classes,batchnorm):
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = UpSampling2D()(c5)
    u6 = Conv2D(filters = n_filters *8, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u6)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = UpSampling2D()(c6)
    u7 = Conv2D(filters = n_filters *4, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u7)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = UpSampling2D()(c7)
    u8 = Conv2D(filters = n_filters *2, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u8)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = UpSampling2D()(c8)
    u9 = Conv2D(filters = n_filters *1, kernel_size = (2, 2), kernel_initializer = 'he_normal', padding = 'same')(u9)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(classes, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model

class Conv2D_custom(layers.Layer):
    
    ''' args:

            filters: Integer, the number of filters in convolution
            prefix: String, name of the block of which this layer is a part
            stride: Integer, stride for convolution, default=1 
            kernel_size: Integer, size of kernel for convolution, default=3
            rate: Integer, atrous rate for convolution, default=1

        Input: 4D tensor with shape (batch, rows, cols, channels)
        Output: 4D tensor with shape (batch, new_rows, new_cols, filters)
    '''

    def __init__(self, filters, prefix='', stride=1, kernel_size=3, rate=1):
        super(Conv2D_custom, self).__init__()

        self.stride = stride
        # manual padding when stride!=1
        if stride!=1:
            #effective kernel size = kernel_size + (kernel_size - 1) * (rate - 1)
            n_pads = (kernel_size + (kernel_size - 1) * (rate - 1) - 1) // 2
            self.zeropad = layers.ZeroPadding2D(padding=n_pads)

        self.conv_2d = layers.Conv2D(filters, kernel_size=kernel_size, strides= stride, dilation_rate=rate,
                            padding='same' if stride==1 else 'valid', name=prefix + 'Conv2D_custom')
    
    def call(self, x):
        if self.stride != 1:
            x = self.zeropad(x)

        x = self.conv_2d(x)
        return x

class SeparableConv_BN(Model):

    ''' Separable convolutions consist in first performing a depthwise spatial convolution
        (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels.

        - This is a custom implementation of SeparableConv2D layer.
        - To be used in encoder (a modified Xception block) of DeeplabV3+.
        
        The difference between this implementation and tf.keras.layers.SeparableConv2D is that
        here, extra batch normalization and ReLU are added after each 3×3 depthwise convolution
        
        args:

            filters: Integer, the number of output filters in pointwise convolution
            prefix: String, name of the block of which this layer is a part
            stride: Integer, stride for depthwise convolution, default=1 
            kernel_size: Integer, size of kernel for depthwise convolution, default=3
            rate: Integer, atrous rate for depthwise convolution, default=1
            depth_activation: Bool, flag to use activation between depthwise & pointwise convolutions

        Input: 4D tensor with shape (batch, rows, cols, channels)
        Output: 4D tensor with shape (batch, new_rows, new_cols, filters)
    '''

    def __init__(self, filters, prefix='', stride=1, kernel_size=3, rate=1, depth_activation=False):
        super(SeparableConv_BN, self).__init__()
        
        self.stride = stride
        self.depth_activation = depth_activation
        # manual padding size when stride!=1
        if stride!=1:
            #effective kernel size = kernel_size + (kernel_size - 1) * (rate - 1)
            n_pads = (kernel_size + (kernel_size - 1) * (rate - 1) - 1) // 2
            self.zeropad = layers.ZeroPadding2D(padding=n_pads)

        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides= stride, dilation_rate=rate,
                            padding='same' if stride==1 else 'valid', name=prefix + '_depthW')
        self.batchnorm_d = layers.BatchNormalization(name=prefix + '_depthW_BN')

        self.pointwise_conv = layers.Conv2D(filters, kernel_size=1, padding='same', name=prefix + '_pointW')
        self.batchnorm_p = layers.BatchNormalization(name=prefix + '_pointW_BN')

    def call(self, x):
        if self.stride != 1:
            x = self.zeropad(x)
        
        if not self.depth_activation:
            x = tf.nn.relu(x)

        x = self.depthwise_conv(x)
        x = self.batchnorm_d(x)
        if self.depth_activation:
            x = tf.nn.relu(x)
        x = self.pointwise_conv(x)
        x = self.batchnorm_p(x)
        if self.depth_activation:
            x = tf.nn.relu(x)
        
        return x
class Xception_Block(Model):
    ''' Basic building block of DeepLabV3+ encoder (modified Xception) network.
        It consists of 3 SeparableConv_BN layers.
        
        args:
            depth_list: list of 3 Integers, number of filters in each SeparableConv_BN. 
            prefix: String, prefix before name of the layer
            short_path_type: String, one of {'conv','sum'} default=None; type of shortcut connection between input and output of the block
            stride: Integer, stride for depthwise convolution in last(3rd) layer
            rate: Integer, atrous rate for depthwise convolution
            depth_activation: Bool, flag to use activation between depthwise & pointwise convolutions
            return_skip: Bool, flag to return additional tensor after 2 SepConvs for decoder
    '''
    def __init__(self, depth_list, prefix='', residual_type=None, stride=1, rate=1, depth_activation=False, return_skip=False):
        super(Xception_Block, self).__init__()

        self.sepConv1 = SeparableConv_BN(filters=depth_list[0], prefix=prefix +'_sepConv1', stride=1, rate=rate, depth_activation=depth_activation)
        self.sepConv2 = SeparableConv_BN(filters=depth_list[1], prefix=prefix +'_sepConv2', stride=1, rate=rate, depth_activation=depth_activation)
        self.sepConv3 = SeparableConv_BN(filters=depth_list[2], prefix=prefix +'_sepConv3', stride=stride, rate=rate, depth_activation=depth_activation)

        if residual_type == 'conv':
            self.conv2D = Conv2D_custom(depth_list[2], prefix=prefix+'_conv_residual', stride=stride, kernel_size=1, rate=1)
            self.batchnorm_res = layers.BatchNormalization(name=prefix + '_BN_residual')

        self.return_skip = return_skip
        self.residual_type = residual_type
    
    def call(self, x):
        output = self.sepConv1(x)
        output = self.sepConv2(output)
        skip = output # skip connection to decoder
        output= self.sepConv3(output)

        if self.residual_type == 'conv':
            res = self.conv2D(x)
            res = self.batchnorm_res(res)
            output += res
        elif self.residual_type == 'sum':
            output +=  x
        else:
            if(self.residual_type):
                raise ValueError('Arg residual_type should be one of {conv, sum}')
        
        if self.return_skip:
            return output, skip

        return output
# DeepLabV3+ model

class DeepLabV3plus(Model):
    

    def __init__(self, input_size=(512, 512, 3), n_classes=4):
        super(DeepLabV3plus, self).__init__()
        
        self.n_classes = n_classes
        self.input_size = input_size

        # Encoder block
        self.conv2d1 = layers.Conv2D(32, (3, 3), strides=2, name='entry_conv1', padding='same')
        self.bn1 = layers.BatchNormalization(name='entry_BN')
        self.custom_conv1 = Conv2D_custom(64, kernel_size=3, stride=1, prefix='entry_conv2')
        self.bn2 = layers.BatchNormalization(name='conv2_s1_BN')

        self.entry_xception1 = Xception_Block([128, 128, 128], prefix='entry_x1', residual_type='conv', stride=2, rate=1)
        self.entry_xception2 = Xception_Block([256, 256, 256], prefix='entry_x2', residual_type='conv', stride=2, rate=1, return_skip=True)
        self.entry_xception3 = Xception_Block([728, 728, 728], prefix='entry_x3', residual_type='conv', stride=2, rate=1)

        self.middle_xception = [Xception_Block([728, 728, 728], prefix=f'middle_x{i+1}', residual_type='sum', stride=1, rate=1) for i in range(16)]

        self.exit_xception1 = Xception_Block([728, 1024, 1024], prefix='exit_x1', residual_type='conv', stride=1, rate=1)
        self.exit_xception2 = Xception_Block([1536, 1536, 2048], prefix='exit_x2', residual_type=None, stride=1, rate=2, depth_activation=True)

        # Feature projection
        self.conv_feat = layers.Conv2D(256, (1, 1), padding='same', name='conv_featureProj')
        self.bn_feat = layers.BatchNormalization(name='featureProj_BN')
        self.atrous_conv1 = SeparableConv_BN(filters=256, prefix='aspp1', stride=1, rate=6, depth_activation=True)
        self.atrous_conv2 = SeparableConv_BN(filters=256, prefix='aspp2', stride=1, rate=12, depth_activation=True)
        self.atrous_conv3 = SeparableConv_BN(filters=256, prefix='aspp3', stride=1, rate=18, depth_activation=True)
        self.image_pooling = layers.AveragePooling2D(8)
        self.conv_pool = layers.Conv2D(256, (1, 1), padding='same', name='conv_imgPool')
        self.bn_pool = layers.BatchNormalization(name='imgPool_BN')
        self.concat1 = layers.Concatenate()
        self.encoder_op = layers.Conv2D(256, (1, 1), padding='same', name='conv_encoder_op')
        self.bn_enc = layers.BatchNormalization(name='encoder_op_BN')

        # Decoder block
        self.upsample1 = layers.UpSampling2D(size=4)
        self.conv_low = layers.Conv2D(48, (1, 1), padding='same', name='conv_lowlevel_f')
        self.bn_low = layers.BatchNormalization(name='low_BN')
        self.concat2 = layers.Concatenate()
        self.sepconv_last = SeparableConv_BN(filters=256, prefix='final_sepconv', stride=1, depth_activation=True)
        
        self.out_conv = layers.Conv2D(self.n_classes, (1, 1), activation='sigmoid', padding='same', name='output_layer')
        self.upsample2 = layers.UpSampling2D(size=4)

    def call(self, inputs):
        #===================#
        #  Encoder Network  #
        #===================#
        # Entry Block
        x = self.conv2d1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.custom_conv1(x)
        x = self.bn2(x)

        x = self.entry_xception1(x)
        x, skip1 = self.entry_xception2(x)
        x = self.entry_xception3(x)

        # Middle Block
        for i in range(16):
            x = self.middle_xception[i](x)

        # Exit Block
        x = self.exit_xception1(x)
        x = self.exit_xception2(x)

        #====================#
        # Feature Projection #
        #====================#

        b0 = self.conv_feat(x)
        b0 = self.bn_feat(b0)
        b0 = tf.nn.relu(b0)

        b1 = self.atrous_conv1(x)
        b2 = self.atrous_conv2(x)
        b3 = self.atrous_conv3(x)

        # Image Pooling
        b4 = self.image_pooling(x)
        b4 = self.conv_pool(b4)
        b4 = self.bn_pool(b4)
        b4 = tf.nn.relu(b4)
        b4 = tf.image.resize(b4, size=[b3.get_shape()[1], b3.get_shape()[2]])
        

        x = self.concat1([b4, b0, b1, b2, b3])

        x = self.encoder_op(x)
        x = self.bn_enc(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, rate=0.1)

        #===================#
        #  Decoder Network  #
        #===================#

        x = self.upsample1(x)

        low_level = self.conv_low(skip1)
        low_level = self.bn_low(low_level)
        low_level = tf.nn.relu(low_level)
        x = self.concat2([x, low_level])

        x = self.sepconv_last(x)

        x = self.out_conv(x)
        x = self.upsample2(x)
        return x