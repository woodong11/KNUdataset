import os
import cv2
import argparse
import matplotlib
import numpy as np

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from tensorflow.keras.models import load_model
from layers import BilinearUpSampling2D
from matplotlib import pyplot as plt

# input image preprocessing
def load_and_process_image(img):
    # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # in_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    in_image = cv2.imread(img)

    in_image = in_image / 255.0
    return in_image



def path_to_image(true_path, pred_path):
    # true images
    true_image = np.load(true_path)                             # load npy file
    true_image = true_image.astype(np.float32)
    true_image = cv2.resize(true_image, (320, 240))
    
    # pred images
    inputs = np.array(load_and_process_image(pred_path))
    inputs = np.expand_dims(inputs, axis=0)                 # Add a batch dimension
    
    '''you must choose one of the following'''
    # outputs = nyu_predict(model, inputs, minDepth=0, maxDepth=1000, batch_size=2)       # when using nyu.h5
    outputs = model.predict(inputs, batch_size=1)                                     # when using output
    ''''''

    pred_image = np.squeeze(outputs, axis=0)                   # Remove batch dimension
    pred_image = np.squeeze(pred_image, axis = -1) 

    return true_image, pred_image

# calculate accuracy(delta)
def calculate_accuracy(true_image, pred_image):
    thres = np.maximum((true_image/pred_image), (pred_image/true_image))
    delta1 = (thres < 1.25).mean()
    delta2 = (thres < 1.25**2).mean()
    delta3 = (thres < 1.25**3).mean()
    return delta1, delta2, delta3 


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
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) /100



# Path, tolerance
model_path = './weights/rgb_2_50_9acc.h5'
# model_path = 'nyu.h5'
input_folder = './knudataset_900test/test/thermal'
true_folder = './knudataset_900test/test/depth'
delta = 1
image_number = 100



# Load model
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
print('Loading model...')
model = load_model(model_path, custom_objects=custom_objects, compile=False)
print('\nModel loaded ({0}).'.format(model_path))


# Initialize total accuracy to 0
total_accuracy_delta1 = 0
total_accuracy_delta2 = 0
total_accuracy_delta3 = 0

# For each image in dataset
for i in range(1, 101): # images number from 0001 to 0100
    pred_path = os.path.join(input_folder, f'night_thermal_{i:04d}.png')
    true_path = os.path.join(true_folder, f'night_depth_{i:04d}.npy')
    print(pred_path)

    # Load and preprocess data
    true_image, pred_image = path_to_image(true_path, pred_path)
    cv2.imshow('pred', pred_image)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()

    # Calculate accuracy
    accuracy_delta1, accuracy_delta2, accuracy_delta3 = calculate_accuracy(true_image, pred_image)
    total_accuracy_delta1 += accuracy_delta1
    total_accuracy_delta2 += accuracy_delta2
    total_accuracy_delta3 += accuracy_delta3
    print(f'Image {i:04d} Accuracy delta1: {accuracy_delta1 * 100}%')
    print(f'Image {i:04d} Accuracy delta2: {accuracy_delta2 * 100}%')
    print(f'Image {i:04d} Accuracy delta3: {accuracy_delta3 * 100}%')

# Calculate average accuracy
average_accuracy_delta1 = total_accuracy_delta1 / image_number
average_accuracy_delta2 = total_accuracy_delta2 / image_number
average_accuracy_delta3 = total_accuracy_delta3 / image_number
print(f'Average Accuracy delta1: {average_accuracy_delta1 * 100}%')
print(f'Average Accuracy delta2: {average_accuracy_delta2 * 100}%')
print(f'Average Accuracy delta3: {average_accuracy_delta3 * 100}%')


#matplotlib problem on ubuntu terminal fix
matplotlib.use('TkAgg')
plt.imshow(pred_image)
plt.show()









