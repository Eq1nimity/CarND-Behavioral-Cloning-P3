"""
Reads data from sample folders. 

Augments and combines into single output folder.

This allows for future testing, as the state of the original data is unaltered. At cost of disk space.
"""
import csv
from numpy import repeat, asarray

DATA_CSV_PATH = './data/driving_log.csv'
SUPPLEMENTARY_DATA_CSV_PATH = './data/supplementary-data/driving_log.csv'

OUTFILE = 'combined_data.csv'


if __name__ == '__main__':
    data = import_csv(DATA_CSV_PATH)






# def process_image_data_generator(csv_lines, img_path, multi_cam=True):
#     """
#     Generate image data.
#     :param csv_lines:
#     :param img_path:
#     :return:
#     """
#     images = []
#     for line in csv_lines[1:]:  # Skip header/ first data entry
#         center, left, right = line[0], line[1], line[2]
#         cameras = [center, left, right]
#         for camera in cameras:
#             filename = camera.split('/')[-1]
#             path = img_path + filename
#             yield images.append(cv2.imread(path))
#
# """ The generator will need to be run at a lower level... e.g. it would have to yeild the
# line. The processor would use the generator function inside of it, to run over and over through all
#  iterations. I believe. """

#
# def process_image_data(csv_lines, img_path, multi_cam=True):
#     """
#     :param csv_lines: anticipates import_csv() method
#     :param img_path: path to img directory
#     :param multi_cam: True: Use all camera angles. False: use just center.
#     :return: X_train as np.array of image data.
#     """
#     images = []
#     if multi_cam:
#         for line in csv_lines[1:]:  # Skip header or first data entry.
#             center, left, right = line[0], line[1], line[2]
#             cameras = [center, left, right]
#             for camera in cameras:
#                 filename = camera.split('/')[-1]
#                 path = img_path + filename
#                 images.append(cv2.imread(path))
#
#     else:
#         for line in csv_lines[1:]:
#             source_path = line[0]
#             filename = source_path.split('/')[-1]
#             path = img_path + filename
#             image = cv2.imread(path)
#             images.append(image)
#
#     return images







# Notes:
# 1. Should I run process_image_data funct as a generator? How? Yes.
# 2. What if I run the multiplier prior to loading the images? E.g. manipulated the csv, THEN loaded.