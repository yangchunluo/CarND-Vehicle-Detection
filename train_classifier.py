import os
import numpy as np
import cv2
import time
import argparse
import pickle
from collections import namedtuple

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FeatureExtractParams(namedtuple('FeatureExtractParams',
                                      ['color_space', 'spatial_size', 'hist_bins',
                                       'hog_orient', 'hog_pix_per_cell', 'hog_cell_per_block', 'hog_channels'])):
    """
    Parameters used in feature extraction. Must be in-sync between training and inference.
    """


def get_all_files(root_dir, file_ext=None):
    """
    Walk a directory and get all the files matching the extension.
    :param root_dir: root directory
    :param file_ext: file name extension
    :return: list of file paths
    """
    file_list = []
    for dir_name, _, file_names in os.walk(root_dir):
        if file_ext is not None:
            file_names = filter(lambda f: os.path.basename(f).endswith(file_ext), file_names)
        file_list.extend([os.path.join(dir_name, f) for f in file_names])
    print(len(file_list))
    return file_list


def get_all_features(file_list, params):
    """
    Take a list of file paths and produce all the features
    :param file_list: list of file paths
    :param params: feature extraction parameters
    :return: list of all the features
    """
    feature_list = []
    for filename in file_list:
        img = cv2.imread(filename)
        # if os.path.basename(filename).endswith(".png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        feature_list.append(extract_features(img, params))
    return feature_list


def get_hog_features(img, params):
    """Get HOG features"""
    channel_hogs = [hog(img[:, :, ch], orientations=params.hog_orient,
                        pixels_per_cell=(params.hog_pix_per_cell, params.hog_pix_per_cell),
                        cells_per_block=(params.hog_cell_per_block, params.hog_cell_per_block),
                        transform_sqrt=True,
                        visualise=False, feature_vector=True)
                    for ch in params.hog_channels]
    return np.concatenate(channel_hogs)


def get_binned_spatial_features(img, params):
    """Get binned spatial features"""
    return cv2.resize(img, (params.spatial_size, params.spatial_size)).ravel()


def get_color_histogram_features(img, params):
    """Get color histogram features"""
    # Compute the histogram of the color channels separately.
    channel_hist = [np.histogram(img[:, :, ch], bins=params.hist_bins, range=(0, 256))
                    for ch in range(3)]
    # Concatenate the histograms into a single feature vector.
    return np.concatenate([hist[0] for hist in channel_hist])


def extract_features(image, params):
    """
    Extract features from an image
    :param image: cv2 read-in image, assuming BGR format with pixel values [0-255]
    :param params: feature extraction parameters
    :return: all the features concatenated as vector
    """
    # Color space conversion.
    if params.color_space == 'RGB':
        feature_image = np.copy(image)
    elif params.color_space == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif params.color_space == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif params.color_space == 'HLS':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif params.color_space == 'YUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif params.color_space == 'YCrCb':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        raise ValueError("Invalid color space {}".format(params.color_space))

    spatial_features = get_binned_spatial_features(feature_image, params)
    histogram_features = get_color_histogram_features(feature_image, params)
    hog_features = get_hog_features(feature_image, params)
    # print(len(spatial_features))
    # print(len(histogram_features))
    # print(len(hog_features))

    return np.concatenate((spatial_features, histogram_features, hog_features))


def train_classifier(input_paths, split_portion, params):
    """
    Train a classifier.
    :param input_paths: a tuple of (positive, negative) directory root.
    :param split_portion: train-test split portion
    :param params: feature extraction parameters.
    :return: classifier, feature_scaler
    """
    # Get all the features.
    features = []
    for path, name in zip(input_paths, ["positive", "negative"]):
        t0 = time.time()
        features.append(get_all_features(get_all_files(path, ".png")[:1000], params))
        print('{:.2f} seconds to get {} features'.format(time.time() - t0, name))

    # Feature normalization.
    print("Positive sample size {}".format(len(features[0])))
    print("Negative sample size {}".format(len(features[1])))
    X = np.vstack(features).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    X_scaled = X_scaler.transform(X)

    # Get the labels.
    y = np.hstack((np.ones(len(features[0])),
                   np.zeros(len(features[1]))))

    # Split up data into randomized training and test sets.
    rand_state = 37
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=split_portion, random_state=rand_state)
    print("Training sample size {}".format(len(y_train)))
    print("Test sample size {}".format(len(y_test)))

    # Train the SVM classifier.
    svc = LinearSVC()
    t0 = time.time()
    svc.fit(X_train, y_train)
    t1 = time.time()
    print(round(t1 - t0, 2), 'seconds to train SVC')

    # Test the SVM accuracy
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

    return svc, X_scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive-dir', type=str, required=False, default='./training_images/vehicles',
                        help='Directory of positive sample images (having a vehicle)')
    parser.add_argument('--negative-dir', type=str, required=False, default='./training_images/non-vehicles',
                        help='Directory of negative sample images (not having a vehicle)')
    parser.add_argument('--split-portion', type=float, required=False, default=0.2,
                        help='Train-test split (portion for test)')
    parser.add_argument('--output-file', type=str, required=False, default='./trained-model-params.p',
                        help='Output pickle file for the trained model and its parameters')
    x = parser.parse_args()

    feature_params = FeatureExtractParams(color_space="RGB", spatial_size=32, hist_bins=32,
                                          hog_orient=8, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channels=[0, 1])
    svc, scaler = train_classifier((x.positive_dir, x.negative_dir), x.split_portion, feature_params)

    # Save the model and its parameters
    dist_pickle = dict()
    dist_pickle["classifier"] = svc
    dist_pickle["feature_scaler"] = scaler
    dist_pickle["feature_params"] = feature_params
    pickle.dump(dist_pickle, open(x.output_file, "wb"))



