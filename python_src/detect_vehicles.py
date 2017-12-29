import os
import cv2
import numpy as np

from itertools import chain
from collections import namedtuple
from scipy.ndimage.measurements import label

from train_classifier import (TRAIN_SAMPLE_SIZE, combine_features, get_hog_features_for_channel,
                              convert_color_space, get_binned_spatial_features, get_color_histogram_features)
from utils import output_img, get_img_size


# Heatmap thresholds for window (current frame) and individual pixel (aggregated over past).
HEATMAP_THRESHOLD_WINDOW = 1.3  # Current confidence score
HEATMAP_THRESHOLD_WINDOW_PORTION = 0.5  # Percentage
HEATMAP_THRESHOLD_PIXEL = 1  # Aggregated confidence score


class VehicleDetectionParams(namedtuple("VehicleDetectionParams",
                             ["classifier", "feature_scaler", "feature_params"])):
    """Parameters for the vehicle detection model"""


class SearchOptions(namedtuple("SearchOptions",
                               ["window_scale", "y_start", "y_stop", "min_confidence", "cell_per_step"])):
    """Search window options"""


class VehicleHistoryInfo:
    """History information for vehicle detection"""
    def __init__(self):
        # Exponential decay of heatmap
        self.heatmap = None

        # Constant parameters
        self.decay_rate = 0.2


def draw_windows(img, window_list):
    """Draw the boxes on image"""
    imcopy = np.copy(img)
    # Use different color for different window size
    color_size_boundary = [80, 100]
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    assert len(colors) == len(color_size_boundary) + 1
    for box, confidence in window_list:
        # Determine which color to use given the window size
        size = abs(box[0][0] - box[1][0])
        color_idx = 0
        for b in color_size_boundary:
            if size < b:
                break
            color_idx += 1
        # Draw a rectangle given window coordinates
        cv2.rectangle(imcopy, box[0], box[1], colors[color_idx], 3)
        # Put down the confidence score
        cv2.putText(imcopy, "{:.2f}".format(confidence), org=(box[0][0], box[1][1] + 13),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2, color=(255, 255, 255))
    return imcopy


def precompute_for_all_searches(img, search_opts, params):
    """
    Pre-compute HOG features once this image frame
    :param img: input image
    :param search_opts: all the search options (for min/max Y boundary)
    :param params: feature extraction parameters
    :return: image converted (color space), hog features
    """
    # Get the min and max of Y boundary.
    y_min = np.min([opt.y_start for opt in search_opts])
    y_max = np.max([opt.y_stop for opt in search_opts])
    # Slice the image.
    img_slice = img[y_min:y_max, :, :]
    # Convert color space.
    img_slice = convert_color_space(img_slice, params)
    return img_slice, y_min


def search_scaled_window(precomputed, search_opt, clf, scaler, params):
    """
    Search given the search window, using sub-sampling for HOG features.
    :param precomputed: precomputed stuff (see precompute_hog_on_slice)
    :param search_opt: search window options
    :param clf: pre-trained classifier
    :param scaler: feature scaler
    :param params: feature extraction parameters
    :return: list of windows in which vehicle is detected
    """
    # Extract image slice for this search region.
    img_slice, y_min = precomputed
    img_slice = img_slice[search_opt.y_start - y_min:search_opt.y_stop - y_min, :, :]
    slice_size = get_img_size(img_slice)

    # Scale the sliced region
    if search_opt.window_scale != 1:
        img_slice = cv2.resize(img_slice, (int(slice_size[0] / search_opt.window_scale),
                                           int(slice_size[1] / search_opt.window_scale)))
        slice_size = get_img_size(img_slice)

    def get_nblocks(size):
        return (size // params.hog_pix_per_cell) - params.hog_cell_per_block + 1

    # Get the number of blocks and steps
    nblocks_x = get_nblocks(slice_size[0])
    nblocks_y = get_nblocks(slice_size[1])
    nblocks_window = get_nblocks(TRAIN_SAMPLE_SIZE)
    nsteps_x = (nblocks_x - nblocks_window) // search_opt.cell_per_step + 1
    nsteps_y = (nblocks_y - nblocks_window) // search_opt.cell_per_step + 1

    # Compute individual channel HOG features.
    # Don't seem to be able to pre-compute this (possibly due to different window size at different scale).
    hog_slice = [get_hog_features_for_channel(img_slice, ch, params)
                 for ch in params.hog_channels]

    positive_windows = []
    for xb in range(nsteps_x):
        for yb in range(nsteps_y):
            ypos = yb * search_opt.cell_per_step
            xpos = xb * search_opt.cell_per_step

            # Extract HOG features
            hog_subsampled = [hc[ypos:ypos + nblocks_window, xpos:xpos + nblocks_window].ravel()
                              for hc in hog_slice]

            # Extract original image patch and get other features
            xleft = xpos * params.hog_pix_per_cell
            ytop = ypos * params.hog_pix_per_cell
            subimg = img_slice[ytop:ytop + TRAIN_SAMPLE_SIZE, xleft:xleft + TRAIN_SAMPLE_SIZE]

            # Combine all the features and scale them
            all_features = combine_features(spatial_features=get_binned_spatial_features(subimg, params),
                                            histogram_features=get_color_histogram_features(subimg, params),
                                            hog_features=np.hstack(hog_subsampled))
            test_features = scaler.transform(all_features.reshape(1, -1))

            # Make prediction
            if clf.predict(test_features) == 1:
                confidence = clf.decision_function(test_features)[0]
                if confidence < search_opt.min_confidence:
                    continue
                xleft_orig = int(xleft * search_opt.window_scale)
                ytop_orig = int(ytop * search_opt.window_scale)
                winsize_orig = int(TRAIN_SAMPLE_SIZE * search_opt.window_scale)
                window = ((xleft_orig, ytop_orig + search_opt.y_start),
                          (xleft_orig + winsize_orig, ytop_orig + winsize_orig + search_opt.y_start))
                positive_windows.append((window, confidence))

    return positive_windows


def vehicle_detection_pipeline(img, vehicle_detection_params, vehicle_hist, output_dir, img_base_fname):
    """
    Image processing pipeline.
    :param img: image pixels array, in RGB color space
    :param vehicle_detection_params: needed parameters
    :param vehicle_hist: vehicle detection history
    :param output_dir: intermediate output directory
    :param img_base_fname: base filename for this image, None for disabling intermediate output
    :return: see below
    """
    # Search through a list of options.
    search_opt = [
        # SearchOptions(window_scale=0.5, y_start=400, y_stop=500, min_confidence=0.1, cell_per_step=2),
        SearchOptions(window_scale=1.0, y_start=400, y_stop=600, min_confidence=0.1, cell_per_step=2),
        SearchOptions(window_scale=1.5, y_start=400, y_stop=656, min_confidence=0.1, cell_per_step=2),
        SearchOptions(window_scale=2.0, y_start=400, y_stop=680, min_confidence=0.1, cell_per_step=2),
        # SearchOptions(window_scale=3.5, y_start=400, y_stop=680, min_confidence=0.1, cell_per_step=2),
    ]
    # Precompute once, instead of in each search
    precomputed = precompute_for_all_searches(img, search_opt, vehicle_detection_params.feature_params)

    positive_windows = list(chain.from_iterable(  # Essentially a flatmap operation
        map(lambda opt: search_scaled_window(precomputed, opt, vehicle_detection_params.classifier,
                                             vehicle_detection_params.feature_scaler,
                                             vehicle_detection_params.feature_params),
            search_opt)))
    searchbox_display = draw_windows(img, positive_windows)
    if img_base_fname is not None:
        output_img(searchbox_display, os.path.join(output_dir, "searchbox", img_base_fname))

    # Build the current heat map using confidence score for each search window.
    heatmap = np.zeros(img.shape[:2]).astype(np.float)
    for box, confidence in positive_windows:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += confidence
    # For each search window, count the portion of high-confidence pixels.
    # If below a threshold, remove this low-quality window.
    heatmap_copy = np.copy(heatmap)
    for box, confidence in positive_windows:
        roi = heatmap_copy[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        above_count = np.count_nonzero(roi >= HEATMAP_THRESHOLD_WINDOW)
        if above_count / float(roi.shape[0] * roi.shape[1]) < HEATMAP_THRESHOLD_WINDOW_PORTION:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] -= confidence

    # Update vehicle history *before* thresholding individual pixels.
    if vehicle_hist is not None:
        # First frame.
        if vehicle_hist.heatmap is None:
            vehicle_hist.heatmap = np.copy(heatmap)  # IMPORTANT: must deep copy
        # Exponential decay update.
        vehicle_hist.heatmap *= (1 - vehicle_hist.decay_rate)
        vehicle_hist.heatmap += heatmap * vehicle_hist.decay_rate
        del heatmap
        # For the rest of the function. Must deep copy.
        heatmap = np.copy(vehicle_hist.heatmap)

    # Increase the heatmap magnitude for visualization.
    boost = 170
    heatmap_display = np.dstack((np.clip(heatmap * boost, 0, 255),  # R
                                 np.zeros_like(heatmap),            # G
                                 np.zeros_like(heatmap)))           # B
    cv2.putText(heatmap_display, "Threshold intensity:", org=(70, 110),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3, color=(255, 255, 255))
    heatmap_display[40:140, 700:800, 0] = HEATMAP_THRESHOLD_PIXEL * boost
    if img_base_fname is not None:
        output_img(heatmap_display, os.path.join(output_dir, "heatmap", img_base_fname))

    # Filter individual pixels.
    heatmap[heatmap < HEATMAP_THRESHOLD_PIXEL] = 0

    # Label individual vehicles
    labels = label(heatmap)
    vehicles = []
    for obj_number in range(labels[1]):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == obj_number + 1).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        vehicles.append(bbox)

    return vehicles, searchbox_display, heatmap_display
