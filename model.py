import csv
import cv2
from numpy import repeat, array, append
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from tensorflow import reset_default_graph
reset_default_graph()

DATA_CSV_PATH = './data/driving_log.csv'
DATA_IMG_PATH = './data/IMG/'
SUPPLEMENTARY_DATA_CSV_PATH = './data/supplementary-data/driving_log.csv'
SUPPLEMENTARY_DATA_IMG_PATH = './data/supplementary-data/IMG/'
COMBINED_DATA_IMG_PATH = './data/combined/IMG'

OUT_FILE = 'model2.h5'
MULTI_CAM_ANGLE_CORRECTION =  0.33 # (left: -1 , right: 1)
MULTI_CAM = True

# Preprocess data functions

def import_csv(csv_path):
    """
    :param csv_path: input DATA_CSV_PATH
    :return: list of lists (csv lines).
    """
    lines = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        [lines.append(line) for line in reader]
    return array(lines)


def augment_csv(csv_lines, factor):
    """
    Multiplies input csv_lines by specified factor. Ensures row persistence by
    nesting, and then repeating over the 0th axis.
    :param csv_lines: single csv_lines array
    :param factor: multiplication factor.
    :return: a ndarray of length csv_lines * factor.
    """
    csv_lines = repeat(csv_lines, factor, axis=0)
    return array(csv_lines)

def combine_data(orig_data, aug_data):
    """
    Combines input arrays into single array.

    :param csv_lines: n number of csv_line arrays.
    :return: ndarray with input arrays concatenated.
    """
    return append(orig_data, aug_data, axis=0)


# Load data functions

def process_image_data(csv_lines, img_path):
    """
    :param csv_lines: anticipates import_csv() method
    :param img_path: path to img directory
    :param crop_images:
    :param multi_cam: True: Use all camera angles. False: use just center.
    :return: X_train as np.array of image data.
    """
    images = []
    for line in csv_lines[1:]:  # Skip header/ first data entry
        center, left, right = line[0], line[1], line[2]
        cameras = [center, left, right]
        for camera in cameras:
            filename = camera.split('/')[-1]
            path = img_path + '/' + filename
            images.append(cv2.imread(path))

    return array(images)

def process_measurement_data(csv_lines):
    """

    :param csv_lines: input
    :return: y_train as np.array of measurement data.
    """
    measurements = []
    for line in csv_lines[1:]:
        center = line[3]
        left = float(line[3]) + MULTI_CAM_ANGLE_CORRECTION
        right = float(line[3]) + -MULTI_CAM_ANGLE_CORRECTION    # Note: Negative Correction
        angles = [center, left, right]
        [measurements.append(angle) for angle in angles]


    return array(measurements)


Model functions

def train_model(X_train, y_train, outfile):

    input_shape = (160,320,3)

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    adam = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, y_train, verbose=2, shuffle=False, epochs=5)

    model.save(outfile)

# Abstracted functions

def load_and_combine_data():
    data = import_csv(DATA_CSV_PATH)
    sup_data = import_csv(SUPPLEMENTARY_DATA_CSV_PATH)
    sup_data = augment_csv(sup_data, 3)
    return combine_data(data, sup_data)

# Final
#  if __name__ == '__main__':
#     combined_data = load_and_combine_data()
#     X_train = process_image_data(combined_data, COMBINED_DATA_IMG_PATH)
#     y_train = process_measurement_data(combined_data)
#     train_model(X_train, y_train, OUT_FILE)

if __name__ == '__main__':
    data = import_csv(DATA_CSV_PATH)
    X_train = process_image_data(data, DATA_IMG_PATH)
    y_train = process_measurement_data(data)
    train_model(X_train, y_train, OUT_FILE)