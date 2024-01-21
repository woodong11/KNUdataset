#-*-coding:utf-8-*-
import sys
import argparse
import os
import glob
import cv2
import numpy as np
import random
import tensorflow as tf
from data import get_nyu_train_test_data, get_unreal_train_test_data
# from callbacks import get_nyu_callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Concatenate, UpSampling2D
from tensorflow.keras.applications import DenseNet169, DenseNet201
import tensorflow.keras.backend as K

sample_number = 800

# TensorFlow GPU 메모리 사용 제한 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 사용 제한 설정
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # 사용할 GPU 지정
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("########### using GPU ###############")
    except RuntimeError as e:
        print("############## GPU 설정 불가 ##############")

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=10.0):
    print("###########y_true shape:###########", tf.shape(y_true), y_true.shape)
    print("###########y_pred shape:###########", tf.shape(y_pred), y_pred.shape)
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Modify this line
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
    l_ssim = K.clip((1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, maxDepthVal))) * 0.5, 0, 1)
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


def create_model(existing='', is_twohundred=False, is_halffeatures=True):
    if len(existing) == 0:
        if is_twohundred:
            base_model = DenseNet201(include_top=False, weights='imagenet')
        else:
            base_model = DenseNet169(include_top=False, weights='imagenet')
        
        ''''''

        ''''''
        
        base_model_output_shape = base_model.layers[-1].output.shape
        for layer in base_model.layers:
            layer.trainable = True
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1])/2)
        else:
            decode_filters = int(base_model_output_shape[-1])
        def upproject(tensor, filters, name, concat_with):
            up_i = UpSampling2D(size=(2, 2), name=name+'_upsampling2d')(tensor)  # UpSampling2D to (2, 2)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output])
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i
        

        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', name='conv2')(base_model.output)
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False:
            decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)
        model = Model(inputs=base_model.input, outputs=conv3)
    else:
        custom_objects = {'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
    return model


def get_knu_data(batch_size):
    input_files = sorted(glob.glob("./knudataset_900test/day/thermal/*.png"))
    output_files = sorted(glob.glob("./knudataset_900test/day/depth/*.npy"))

    def process_filepaths(in_filepath, out_filepath):
        img = cv2.imread(in_filepath, cv2.IMREAD_GRAYSCALE)
        #img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        in_image = (img / 255.0).astype(np.float32)  
        #in_image = cv2.resize(in_image, (640, 480)) 
        
        out_depth = np.load(out_filepath)
        out_depth = (out_depth).astype(np.float32)  # Convert to float32
        out_depth = cv2.resize(out_depth, (320, 240))
        out_depth = np.expand_dims(out_depth, axis=-1)
        return in_image, out_depth

    print(len(input_files))
    print(len(output_files))

    while True:
        combined = list(zip(input_files, output_files))
        random.shuffle(combined)
        input_files[:], output_files[:] = zip(*combined)
        for i in range(0, len(input_files), batch_size):
            batch_input_files = input_files[i:i+batch_size]
            batch_output_files = output_files[i:i+batch_size]
            batch_x = []
            batch_y = []
            for in_filepath, out_filepath in zip(batch_input_files, batch_output_files):
                x, y = process_filepaths(in_filepath, out_filepath)
                batch_x.append(x)
                batch_y.append(y)

            # batch_x = np.random.rand(batch_size, 480, 640, 3)
            # batch_y = np.random.rand(batch_size, 480, 640, 1)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            yield batch_x, batch_y


def get_knu_data_test(batch_size):
    input_files = sorted(glob.glob("./knudataset_900test/night/thermal/*.png"))
    output_files = sorted(glob.glob("./knudataset_900test/night/depth/*.npy"))

    def process_filepaths(in_filepath, out_filepath):
        img = cv2.imread(in_filepath, cv2.IMREAD_GRAYSCALE)
        # img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        in_image = (img / 255.0).astype(np.float32)  
        #in_image = cv2.resize(in_image, (640, 480)) 

        out_depth = np.load(out_filepath)
        out_depth = (out_depth).astype(np.float32) 
        out_depth = cv2.resize(out_depth, (320, 240))
        out_depth = np.expand_dims(out_depth, axis=-1)
        return in_image, out_depth

    while True:
        for i in range(0, len(input_files), batch_size):
            batch_input_files = input_files[i:i+batch_size]
            batch_output_files = output_files[i:i+batch_size]
            batch_x = []
            batch_y = []
            for in_filepath, out_filepath in zip(batch_input_files, batch_output_files):
                x, y = process_filepaths(in_filepath, out_filepath)
                batch_x.append(x)
                batch_y.append(y)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            yield batch_x, batch_y

################# Command line arguments ########################   
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--checkpoint', default='', type=str, help='Checkpoint/model to resume from')
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
parser.add_argument('--data', default='nyu', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=2, help='Batch size')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=0.0, help='Minimum of input depths')      # 0m
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')   # 10m
parser.add_argument('--name', type=str, default='densedepth_knu', help='A name to attach to the training session')
parser.add_argument('--full', dest='full', action='store_true', help='Full model with skip connections')
parser.set_defaults(full=False)
args = parser.parse_args()

################ Create the depth prediction model ######################
model = create_model(args.checkpoint, is_twohundred=args.full)
# Training session details
model.compile(optimizer=Adam(lr=args.lr), loss=depth_loss_function, metrics=['accuracy'])
# model.summary() 

# Create a directory to save weights (if it doesn't exist)
os.makedirs("weights", exist_ok=True)


################# Prepare dataset-specific parameters ###################
# train_generator, test_generator = get_nyu_train_test_data( args.bs )    # when using nyu
train_generator = get_knu_data(args.bs)                                   # when using knu_dataset
test_generator = get_knu_data_test(args.bs)

print('Ready for training!')



################## Start training ########################
#callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, sample_number // args.bs),
    epochs=args.epochs,
    callbacks=[
        ModelCheckpoint('./weights/teneth_weights.h5', monitor='loss', save_best_only=True)
        #EarlyStopping(monitor='loss', patience=5)
    ],
    validation_data=test_generator,
    validation_steps=50,
    verbose=1
)

################### evaluate ############################
print("######## Evaluate ########")
number_of_test_samples = 100
scores = model.evaluate(test_generator, steps=number_of_test_samples//args.bs)
print("Metrics Names: ", model.metrics_names)
print("Scores: ", scores)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

################### print model ############################

import matplotlib.pyplot as plt
plt.figure(figsize=(13, 5))
fig, axs = plt.subplots(1, 2)
fig.tight_layout()

acc= history.history['accuracy']
val_acc= history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

 
ax1 = plt.subplot(1,2,1)
plt.title('Loss Graph')
ax1.plot(y_loss, c="blue", label='Trainset_loss')
ax1.plot(y_vloss, c="cornflowerblue", label='Testset_loss')
ax1.legend(['train_loss', 'val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')


ax2 = plt.subplot(1,2,2)
plt.title('Accuracy Graph')
ax2.plot(acc, c="red", label='Trainset_acc')
ax2.plot(val_acc, c="lightcoral", label='Testset_acc')
ax2.legend(['train_acc', 'val_acc'])
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
plt.show()


