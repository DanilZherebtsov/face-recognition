# USE PYTHON 3.7 ENVIRONMENT
import os
import time
import cv2  
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

print(tf.__version__) # 2.7

def load_saved_model():
    '''Load the saved model from the disk'''
    json_file = open('keras-facenet-h5/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('keras-facenet-h5/model.h5')
    return model


def img_to_encoding():
    '''Converts an image to an embedding vector by using the model'''
    ...


def initialize_database():
    '''Initialize the database of people names and their photos encodings'''
    ...


def get_image_from_camera():
    '''This function captures an image from the camera and returns it as a numpy array.'''


def identify_person():
    '''Compare the picture from the camera to the pictures in the database'''


def recognize_face_from_camera(model):
    '''Main function to execute face recognition'''


def add_new_user_to_database(database, model):
    '''Take picture of new employee, store in employees folder and in database as an embedding'''


# show some pictures of people
tf.keras.preprocessing.image.load_img("employees/Sarah Connor.jpg", target_size=(160, 160))
