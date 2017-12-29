import os
import cv2
import copy
import numpy as np

from utils import get_img_size, output_img, draw_outline


def mask_lane_pixels(img, sobelx_thresh, color_thresh):
    """
    Mask lane pixels using gradient and color space information
    :param img: image pixels array, in RGB color space
    """
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel X on L-channel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold on X gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1

    # Threshold on S-channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1

    # Stack each channel for visualization
    masked_color = np.dstack((np.zeros_like(sx_binary),  # Red
                              sx_binary,                 # Green
                              s_binary)                  # Blue
                             ) * 255

    # Combining gradient and color thresholding
    masked = np.zeros_like(s_binary)
    masked[(s_binary == 1) | (sx_binary == 1)] = 255

    return masked, masked_color


def transform_perspective(img, src_pts, dst_pts, img_size):
    """
    Perspective transform
    :param img: image pixels array, in RGB color space
    """
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


def preprocess_image(img, output_dir, img_fname):
    """
    Pre-processing pipeline for an image.
    :param img: image pixels array, in RGB color space
    :param output_dir: output directory
    :param img_fname: output filename for this image, None for disabling output
    :return: undistorted image, masked image in color, warped image, Minv (for warping back)
    """
    img_size = get_img_size(img)  # (width, height)

    # Mask lane pixels
    masked, masked_color = mask_lane_pixels(img, sobelx_thresh=(20, 100), color_thresh=(170, 255))
    if img_fname is not None:
        output_img(masked_color, os.path.join(output_dir, 'masked_color', img_fname))
        output_img(masked, os.path.join(output_dir, 'masked', img_fname))

    # Perspective transform
    # Source points are measured manually from test_images/straight_lines1.jpg by finding a trapezoid.
    src_points = np.float32([
        [576.0, 463.5],  # Top left
        [706.5, 463.5],  # Top right
        [208.0, 720.0],  # Bottom left
        [1095.0, 720.0]  # Bottom right
    ])
    # Destination points are the corresponding rectangle
    dst_points = np.float32([
        [260, 0],
        [980, 0],
        [260, img_size[1]],
        [980, img_size[1]]
    ])
    warped, _, Minv = transform_perspective(masked, src_points, dst_points, img_size)
    if img_fname is not None:
        output_img(warped, os.path.join(output_dir, 'masked-warped', img_fname))

    return masked_color, warped, Minv


class LaneHistoryInfo:
    """History information of lane finding"""
    def __init__(self):
        # Use tracking for the next frame
        self.use_tracking = False
        # Use X base position for detection for the next frame
        self.use_xbase = False
        # Number of continuous frames that failed land finding
        self.n_continuous_failure = 0
        # X base position, smoothed using exponential decay
        self.xbase_position = {}
        # Coefficients of the past fits, smoothed using exponential decay
        self.fit_coeffs = {}
        # Radius of curvature, smoothed using exponential decay
        self.radius_of_curvature = None
        # Distance to center lane, smoothed using exponential decay
        self.distance_to_center = None

        # Constant parameters
        self.decay_rate = 0.2
        self.continuous_failure_threshold = 5


def detect_lane(warped, lane_hist, num_windows, margin_detect, margin_track, recenter_threshold,
                line_distance_threshold, output_dir, img_fname):
    """
    Detect lane and fit curve.
    :param warped: binary warped image
    :param lane_hist: lane history information, set None for separate images
    :param num_windows: number of sliding windows on Y
    :param margin_detect: margin on X for detecting
    :param margin_track: margin on X for tracking
    :param line_distance_threshold: threshold for line distance (upper bound of stdDev, lower bound of mean),
                                    for sanity check
    :param recenter_threshold: a tuple of (t1, t2), if # pixels in the window < t1, recenter window back to base
                               if # pixels in the window > t2, recenter window to the mean the current window
    :param output_dir: output directory
    :param img_fname: output filename for this image, None for disabling output
    :return:
    """
    debug = False

    def poly_value(yval, coeffs):
        return coeffs[0] * yval ** 2 + coeffs[1] * yval + coeffs[2]

    def radius_of_curvature(yval, coeffs):
        return ((1 + (2 * coeffs[0] * yval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    # Constant conversion rate from pixel to world distance
    ym_per_pixel = 30.0 / 720
    xm_per_pixel = 3.7 / 831

    # Create a base color image to draw on and visualize the result
    canvas = cv2.cvtColor(warped.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # Create a color image to draw the lane region
    region = cv2.cvtColor(np.zeros_like(warped).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # Create a color image to draw the lane pixels
    pixels = cv2.cvtColor(np.zeros_like(warped).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    midpoint = np.int(warped.shape[1] / 2)
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    # 1. find pixels that correspond to a lane line
    nonzero_idx = {}
    if lane_hist is not None and lane_hist.use_tracking:
        # Tracking mode: search within a margin of previous fitted curve
        for which in ["left", "right"]:
            nonzeroy_fitx = poly_value(nonzeroy, lane_hist.fit_coeffs[which])
            nonzero_idx[which] = ((nonzerox > (nonzeroy_fitx - margin_track)) &
                                  (nonzerox < (nonzeroy_fitx + margin_track)))

            # Draw search window for the tracking mode
            last_fitx = poly_value(ploty, lane_hist.fit_coeffs[which])
            line_window1 = np.array([np.transpose(np.vstack([last_fitx - margin_track, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([last_fitx + margin_track, ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            window_img = np.zeros_like(canvas)
            cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
            canvas = cv2.addWeighted(canvas, 1, window_img, 0.3, 0)
    else:
        # Detection mode: sliding window
        # X base position will be the starting point for the left and right lines
        xbase = {}
        if lane_hist is not None and lane_hist.use_xbase:
            # Use previously found x base positions
            xbase['left'] = lane_hist.xbase_position['left']
            xbase['right'] = lane_hist.xbase_position['right']
        else:
            # Find the x base position by taking a histogram of the bottom half of the image
            histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
            # Find the peak of the left and right halves of the histogram
            xbase['left'] = np.argmax(histogram[:midpoint])
            xbase['right'] = np.argmax(histogram[midpoint:]) + midpoint

        xcurrent = xbase
        for which in ["left", "right"]:
            all_good_idxs = []
            # Slide through the windows one by one
            window_height = np.int(warped.shape[0] / num_windows)
            for w in range(num_windows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = warped.shape[0] - (w + 1) * window_height
                win_y_high = warped.shape[0] - w * window_height
                win_x_low = xcurrent[which] - margin_detect
                win_x_high = xcurrent[which] + margin_detect

                # Draw the window for visualization
                cv2.rectangle(canvas, (win_x_low, win_y_low), (win_x_high, win_y_high),
                              (0, 255, 0), thickness=2)

                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                # Append these indices to the lists
                all_good_idxs.append(good_inds)

                # If number of good pixels > the threshold, recenter next window on their mean position.
                if len(good_inds) > recenter_threshold[1]:
                    xcurrent[which] = np.int(np.mean(nonzerox[good_inds]))
                    debug and print(w, which, 'updated', xcurrent[which])
                # If number of good pixels < the threshold, recenter next window to base.
                elif len(good_inds) < recenter_threshold[0]:
                    xcurrent[which] = xbase[which]
                    debug and print(w, which, "reverted", xcurrent[which])
                else:
                    debug and print(w, which, 'remained', xcurrent[which])

            # Concatenate the arrays of indices
            nonzero_idx[which] = np.concatenate(all_good_idxs)

    # 2. Fit a polynomial
    coeff_pixel = {}
    coeff_world = {}
    fitx = {}
    for which, color in [("left", [255, 0, 0]), ("right", [0, 0, 255])]:
        # Extract line pixel positions
        lane_x = nonzerox[nonzero_idx[which]]
        lane_y = nonzeroy[nonzero_idx[which]]
        # Color the pixels for visualization
        canvas[lane_y, lane_x] = color
        pixels[lane_y, lane_x] = color

        # Fit a second order polynomial on pixel distance
        coeff_pixel[which] = np.polyfit(lane_y, lane_x, deg=2)
        fitx[which] = poly_value(ploty, coeff_pixel[which])

        # Fit a second order polynomial on world distance
        coeff_world[which] = np.polyfit(lane_y * ym_per_pixel, lane_x * xm_per_pixel, deg=2)

    # 3. Sanity check after both lane lines are fitted.
    xdistances = np.array([lx - rx for lx, rx in zip(fitx['left'], fitx['right'])])
    stddev = np.std(xdistances)
    meandist = abs(np.mean(xdistances))
    debug and print("std_dev:", stddev, "mean:", meandist)
    def coeff_to_str(coeff):
        return "{:.5f} {:.3f} {:.1f}".format(coeff[0], coeff[1], coeff[2])
    cv2.putText(canvas, "StdDev of line distance: {:.1f}".format(stddev), org=(350, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))
    cv2.putText(canvas, "Mean   of line distance: {:.1f}".format(meandist), org=(350, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))
    cv2.putText(canvas, "Left  fit coeff: {}".format(coeff_to_str(coeff_pixel['left'])), org=(350, 150),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))
    cv2.putText(canvas, "Right fit coeff: {}".format(coeff_to_str(coeff_pixel['right'])), org=(350, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))
    success = stddev < line_distance_threshold[0] and meandist > line_distance_threshold[1]
    if not success:
        cv2.putText(canvas, "Sanity check failed", org=(350, 250),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(200, 100, 100))

    # 4. Draw the region between left and right lane lines.
    region_pts = {}
    for which in ['left', 'right']:
        # Visualize the fitted curve on the canvas
        # The equivalent of matplotlib.pyplot.plot(X, Y)
        for x, y in zip(fitx[which], ploty):
            cv2.circle(canvas, center=(int(x), int(y)), radius=3, color=[255, 255, 0], thickness=2)

        # Generate the polygon points to draw the fitted lane region.
        pts = np.transpose(np.vstack([fitx[which], ploty]))
        if which == 'right':
            # So that when h-stacked later, the bottom left lane is adjacent to the bottom right lane (U-shape).
            pts = np.flipud(pts)
        region_pts[which] = np.array([pts])  # Don't miss the [] around pts
    cv2.fillPoly(region, np.int_([np.hstack(region_pts.values())]), (0, 255, 0))

    # 5. Compute radius of curvature and lane X position where the vehicle is (bottom of the view).
    lane_radius = {}
    lane_xpos = {}
    for which in ['left', 'right']:
        curv = radius_of_curvature(np.max(ploty) * ym_per_pixel, coeff_world[which])
        debug and print(which, "curvature", curv)
        lane_radius[which] = curv
        lane_xpos[which] = poly_value(np.max(ploty), coeff_pixel[which])
    # Geometric mean is more stable for radius
    def get_geomean(iterable):
        a = np.log(iterable)
        return np.exp(a.sum() / len(a))
    avg_radius = get_geomean(list(lane_radius.values()))
    dist_center = (midpoint - np.mean(list(lane_xpos.values()))) * xm_per_pixel

    # 6. Update lane history
    if lane_hist is not None:
        # First time update
        if len(lane_hist.fit_coeffs) == 0:
            lane_hist.fit_coeffs = copy.deepcopy(coeff_pixel)
        if len(lane_hist.xbase_position) == 0:
            lane_hist.xbase_position = copy.deepcopy(lane_xpos)
        if lane_hist.radius_of_curvature is None:
            lane_hist.radius_of_curvature = avg_radius
        if lane_hist.distance_to_center is None:
            lane_hist.distance_to_center = dist_center

        if not success:
            lane_hist.n_continuous_failure += 1
            if lane_hist.n_continuous_failure > lane_hist.continuous_failure_threshold:
                lane_hist.use_tracking = False
                lane_hist.use_xbase = False
        else:
            lane_hist.use_tracking = True
            lane_hist.use_xbase = True
            # Exponential decay and update coefficients and xbase positions
            rate = (1 - lane_hist.decay_rate) ** (lane_hist.n_continuous_failure + 1)
            for which in ['left', 'right']:
                lane_hist.xbase_position[which] *= rate
                lane_hist.xbase_position[which] += lane_xpos[which] * (1 - rate)
                lane_hist.fit_coeffs[which] *= rate
                lane_hist.fit_coeffs[which] += coeff_pixel[which] * (1 - rate)
                lane_hist.radius_of_curvature *= rate
                lane_hist.radius_of_curvature += avg_radius * (1 - rate)
                lane_hist.distance_to_center *= rate
                lane_hist.distance_to_center += dist_center * (1 - rate)
            lane_hist.n_continuous_failure = 0

    if img_fname is not None:
        output_img(canvas, os.path.join(output_dir, 'detect-canvas', img_fname))
    return (canvas, region, pixels,
            avg_radius if lane_hist is None else lane_hist.radius_of_curvature,
            dist_center if lane_hist is None else lane_hist.distance_to_center)


def lane_finding_pipeline(img, lane_hist, output_dir, img_base_fname):
    """
    Image processing pipeline.
    :param img: image pixels array, in RGB color space
    :param lane_hist: lane history information for look-ahead filter, reset, and smoothing
    :param output_dir: intermediate output directory
    :param img_base_fname: base filename for this image, None for disabling intermediate output
    :return: see below
    """
    img_size = get_img_size(img)

    # Preprocess image.
    masked_color, warped, Minv = preprocess_image(img, output_dir, img_base_fname)

    # Detect lane.
    window, region, pixels, radius, distance = detect_lane(
        warped, lane_hist, num_windows=16, margin_detect=75, margin_track=75, recenter_threshold=(10, 100),
        line_distance_threshold=(30, 500), output_dir=output_dir, img_fname=img_base_fname)
    draw_outline(window, 10, (192, 192, 192))

    # Warp back.
    lane_region = cv2.warpPerspective(region, Minv, img_size)
    lane_pixels = cv2.warpPerspective(pixels, Minv, img_size)

    return window, lane_region, lane_pixels, radius, distance
