#coding= utf-8
from tensorflow import keras
import tensorflow_datasets as tfds

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="tf_flowers",
                                                            with_info=True,
                                                            split=list(splits),
                                                            # specifying batch_size=-1 will load full dataset in the memory
                                                            #                                                             batch_size=-1,
                                                            # as_supervised: `bool`, if `True`, the returned `tf.data.Dataset`
                                                            # will have a 2-tuple structure `(input, label)`
                                                            as_supervised=True)

IMG_SHAPE=[]

metadata=[]
# Creating a simple CNN model in keras using functional API
def create_model():
    img_inputs = keras.Input(shape=IMG_SHAPE)
    conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(img_inputs)#卷积层
    maxpool_1 = keras.layers.MaxPooling2D((2, 2))(conv_1)#池化层
    conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1)
    maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)
    conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2)
    flatten = keras.layers.Flatten()(conv_3)#扁平化处理了
    dense_1 = keras.layers.Dense(64, activation='relu')(flatten)#全连接层
    output = keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')(dense_1)#输出层

    model = keras.Model(inputs=img_inputs, outputs=output)

    return model
