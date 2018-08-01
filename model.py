import csv
import os
import cv2
import numpy as np
import sklearn

SHOW_HISTORY = False

GEN_FLIP_IMG = True
USE_MULTICAM_IMG = True
SIDE_CAM_ANGLE_CORRECTION = 0.2

ENABLE_CROP = True
CROP_TB = (50, 20)
CROP_LR = (0, 0)
ENABLE_GRAYSCALE = True

BATCH_SIZE = 2

# dataset_path = '{}/behavioral_cloning/driving_dataset'.format(os.getenv('UDACITY_DATASET_PATH'))
dataset_path = '{}/behavioral_cloning/dummy'.format(os.getenv('UDACITY_DATASET_PATH'))


def main():
    train_samples, validation_samples = load_dataset()
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
    train_model(train_generator, validation_generator,
                calc_generated_num(train_samples), calc_generated_num(validation_samples))
    return


def load_dataset():
    """ Return tuple of list of sample data information """
    samples = []
    with open(os.path.join(dataset_path, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    from sklearn.model_selection import train_test_split
    return train_test_split(samples, test_size=0.2)


def calc_generated_num(samples):
    num = len(samples)
    if USE_MULTICAM_IMG:
        num_data_with_multicam = sum(int(has_multicam_image(data)) for data in samples)
        num += 2 * num_data_with_multicam  # adding left and right image data
    if GEN_FLIP_IMG:
        num *= 2
    return num


def has_multicam_image(data):
    # Note: beta_simulator doesn't capture left and right camera image
    return data[1].find('left') != -1 and data[2].find('right') != -1


def generator(samples, batch_size):
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset: offset + batch_size]
            images = []
            angles = []

            def append_data(img_path, angle):
                img_path = img_path.strip()
                filename = img_path.split('\\')[-1]
                current_path = os.path.join(os.path.join(dataset_path, 'IMG'), filename)
                print(current_path)
                image = cv2.imread(current_path)  # BGR order
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(angle)
                return

            for data in batch_samples:
                center_angle = float(data[3])
                append_data(data[0], center_angle)
                if USE_MULTICAM_IMG and has_multicam_image(data):
                    append_data(data[1], center_angle + SIDE_CAM_ANGLE_CORRECTION)  # Left
                    append_data(data[2], center_angle - SIDE_CAM_ANGLE_CORRECTION)  # Right

            augmented_images = []
            augmented_angles = []
            for img, angl in zip(images, angles):
                augmented_images.append(img)
                augmented_angles.append(angl)
                if GEN_FLIP_IMG:
                    augmented_images.append(cv2.flip(img, 1))
                    augmented_angles.append(angl * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def train_model(train_generator, validation_generator, num_train_samples, num_validation_samples):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Cropping2D, Lambda

    print('num train: ', num_train_samples)
    print('num validation: ', num_validation_samples)
    model = Sequential()
    ch = 3
    if ENABLE_CROP:
        model.add(Cropping2D(cropping=(CROP_TB, CROP_LR), input_shape=(160, 320, ch)))
    shape = (160 - (CROP_TB[0] + CROP_TB[1]), 320 - (CROP_LR[0] + CROP_LR[1]), ch)
    # Grayscaling
    if ENABLE_GRAYSCALE:
        # model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))  ## cannot save to pickel
        def rgb2gray(x):  # Stackoverflow #46836358
            return (0.21 * x[:, :, :, :1]) + (0.72 * x[:, :, :, 1:2]) + (0.07 * x[:, :, :, -1:])

        model.add(Lambda(rgb2gray))
        shape = (*shape[:2], 1)

    # Normalize and 0 mean
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=shape, output_shape=shape))

    model.add(Flatten(input_shape=shape))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator,
                                  samples_per_epoch=num_train_samples,
                                  validation_data=validation_generator,
                                  nb_val_samples=num_validation_samples,
                                  nb_epoch=7)
    if SHOW_HISTORY:
        show_history(history)

    model.save('model.h5')
    return


def show_history(history_object):
    import matplotlib.pyplot as plt
    # print the keys contained in the history object
    print(history_object.history.keys())
    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    return

if __name__ == '__main__':
    main()
