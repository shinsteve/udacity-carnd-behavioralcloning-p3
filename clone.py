import csv
import os
import cv2
import numpy as np


def main():
    X_train, y_train = load_dataset()
    train_model(X_train, y_train)
    return


def load_dataset():
    lines = []
#    dataset_path = '{}/behavioral_cloning/driving_dataset'.format(os.getenv('UDACITY_DATASET_PATH'))
    dataset_path = '{}/behavioral_cloning/dummy'.format(os.getenv('UDACITY_DATASET_PATH'))
    with open(os.path.join(dataset_path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        source_path = line[0]
        filename = source_path.split('\\')[-1]
        current_path = os.path.join(os.path.join(dataset_path, 'IMG'), filename)
        # print(current_path)
        image = cv2.imread(current_path)  # BGR order
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


def train_model(X_train, y_train):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense

    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    model.save('model.h5')
    return


if __name__ == '__main__':
    main()
