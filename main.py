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


def img_to_encoding(image_path):
    '''Converts an image to an embedding vector by using the model'''
    img = tf.keras.preprocessing.image.load_img(image_path, target_size = (160,160))
    img = np.around(np.array(img) / 255.0, decimals = 12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def initialize_database():
    '''Initialize the database of people names and their photos encodings'''
    database = {}
    for file in os.listdir('employees'):
        if file.endswith('.jpg'):
            image_path = os.path.join('employees', file)
            embedding = img_to_encoding(image_path)
            database[file[:-4]] = embedding
    return database


def get_image_from_camera(cam_port = 0):
    '''This function captures an image from the camera and returns it as a numpy array.'''
    cam = cv2.VideoCapture(cam_port)
    time.sleep(1)
    result, image = cam.read()
    if result:
        cv2.imshow('Captured image', image)
        cv2.waitKey(2000)
        cv2.destroyWindow('Captured image'); cv2.waitKey(1)
        cam.release()
        return image
    else:
        raise Exception('No image detected. Please try again')


def identify_person(image_path):
    '''Compare the picture from the camera to the pictures in the database'''
    incoming_person_image_encoding = img_to_encoding(image_path)

    distance_between_images = 100

    for name, employee_encoding in database.items():
        dist = np.linalg.norm(incoming_person_image_encoding - employee_encoding)
        if dist < distance_between_images:
            distance_between_images = dist
            identified_as = name
    
    if distance_between_images > 0.7:
        print(f'Not sure, maybe it is {identified_as}')
    else:
        print(f'Employee identified\nName: {identified_as}')
        os.system(f"say 'Hello {identified_as}'")


def recognize_face_from_camera():
    '''Main function to execute face recognition'''
    face_to_recognize = get_image_from_camera()
    cv2.imwrite('face_to_recognize.jpg', face_to_recognize)
    identify_person('face_to_recognize.jpg')
    os.remove('face_to_recognize.jpg')


def add_new_user_to_database():
    '''Take picture of new employee, store in employees folder and in database as an embedding'''
    name = input('Please enter your name: ')
    image = get_image_from_camera()
    image_path = 'employees/' + name + '.jpg'
    cv2.imwrite(image_path, image)
    database[name] = img_to_encoding(image_path)
    print(f'New user "{name}" added to database')
    return database


model = load_saved_model()

database = initialize_database()

database = add_new_user_to_database()

recognize_face_from_camera()