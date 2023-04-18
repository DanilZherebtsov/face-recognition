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


def img_to_encoding(image_path, model):
    '''Converts an image to an embedding vector by using the model'''
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def initialize_database(model):
    '''Initialize the database of people names and their photos encodings'''
    database = {}
    for file in os.listdir("employees"):
        if file.endswith(".jpg"):
            image_path = os.path.join("employees",file)
            database[file[:-4]] = img_to_encoding(image_path, model)
    return database


def get_image_from_camera(cam_port=0):
    '''This function captures an image from the camera and returns it as a numpy array.'''
    
    cam = cv2.VideoCapture(cam_port)
    # give camera time to warm up
    time.sleep(1)
    result, image = cam.read()
    if result:
        cv2.imshow("Captured Image", image)
        cv2.waitKey(1000) # display for 3 seconds
        cv2.destroyWindow("Captured Image"); cv2.waitKey(1)
        cam.release() # turn off camera
        return image
    else:
        raise Exception("No image detected. Please! try again")


def identify_person(image_path, database, model):
    '''Implements face recognition by comparing the image embedding from the camera to the 
    images embeddings in the database.'''
    incoming_person_image_encoding =  img_to_encoding(image_path, model)
    # Initialize "min_dist" to a large value
    distance_between_images = 100
    # Loop over the database dictionary's names and encodings.
    for (name, emplyee_encoding) in database.items():
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(incoming_person_image_encoding - emplyee_encoding)
        if dist < distance_between_images:
            distance_between_images = dist
            identified_as = name
    if distance_between_images > 0.7:
        print(f"Not sure, maybe it is {identified_as}, the distance is {distance_between_images}")
    else:
        print (f"Employee identified\nName: {identified_as}")
        os.system(f"say '{identified_as} recognized'")
    return distance_between_images, identified_as


def recognize_face_from_camera(model):
    '''This function recognizes a face from the camera'''
    face_to_recognize = get_image_from_camera()
    cv2.imwrite('face_to_recognize.jpg', face_to_recognize)
    identify_person('face_to_recognize.jpg', database, model)
    os.remove('face_to_recognize.jpg')


def add_new_user_to_database(database, model):
    '''This function adds a new user to the database by taking a 
    picture from the camera and adding it to the database'''
    name = input("Please enter your name: ")
    image = get_image_from_camera()
    image_path = "employees/" + name + ".jpg"
    cv2.imwrite(image_path, image)
    database[name] = img_to_encoding(image_path, model)
    print(f"New user '{name}' added to database")
    return database


# show some pictures of people
tf.keras.preprocessing.image.load_img("employees/Sarah Connor.jpg", target_size=(160, 160))

FRmodel = load_saved_model()

database = initialize_database(FRmodel)

database = add_new_user_to_database(database, FRmodel)

recognize_face_from_camera(FRmodel)


