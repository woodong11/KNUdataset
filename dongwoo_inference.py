import os
import cv2
import argparse
import matplotlib
import numpy as np
import tensorflow as tf


# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from tensorflow.keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, display_images
from matplotlib import pyplot as plt

def DepthNorm(x, maxDepth):
    return maxDepth / x

# only use when using nyu.h5
def nyu_predict(model, images, minDepth=0, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) /1000



# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='./weights/rgb_2_50_9acc.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='./examples', type=str, help='Input folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Load and preprocess data
thermal_image_files = sorted(os.listdir(args.input))
thermal_image_paths = [os.path.join(args.input, fname) for fname in thermal_image_files]

# 이미지 로드 및 전처리 함수입니다.
def load_and_process_image(thermal_path):
    img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
    #img = cv2.equalizeHist(img)
    in_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    in_image = in_image / 255.0

    return in_image


'''이미지 보기 '''

# 여러개 보고 싶을때
# Input images
inputs = np.array([load_and_process_image(img_path) for img_path in thermal_image_paths])
# inputs = np.array([load_and_process_image(img_path).numpy() for img_path in thermal_image_paths])   # tf 로 받을 경우
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)        # densedepth 논문 버전
# outputs = model.predict(inputs, batch_size=1)    # raw값 보는 버전

#matplotlib problem on ubuntu terminal fix
matplotlib.use('TkAgg')   

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
#plt.savefig('test.png')
plt.show()



# # 하나만 보고 싶을때 
# # Input images
# thermal_image = "./examples/1_image.png"
# inputs = np.array(load_and_process_image(thermal_image))
# inputs = np.expand_dims(inputs, axis=0)  # Add a batch dimension
# print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# # Compute results
# outputs = model.predict(inputs, batch_size=1)
# outputs = model.predict(inputs, batch_size=1)

# # Remove batch dimension
# outputs = np.squeeze(outputs, axis=0)

# #matplotlib problem on ubuntu terminal fix
# matplotlib.use('TkAgg')
# plt.imshow(outputs)
# plt.show()

































