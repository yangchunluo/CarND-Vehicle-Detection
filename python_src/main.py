import os
import cv2
import glob
import pickle
import argparse
import numpy as np
import threading
from moviepy.editor import VideoFileClip

from utils import insert_image, output_img
from train_classifier import FeatureExtractParams
from detect_vehicles import VehicleHistoryInfo, VehicleDetectionParams, vehicle_detection_pipeline
from find_lane_lines import LaneHistoryInfo, lane_finding_pipeline


def fname_generator(max_num_frame=None):
    """Generator for output filename for each frame"""
    idx = 0
    while True:
        idx += 1
        if max_num_frame and idx > max_num_frame:
            yield None  # Stop producing per-frame image output.
        else:
            yield 'video-frame-{}.jpg'.format(idx)


def process_pipeline(img, mtx, dist, vehicle_detection_params, vehicle_hist, lane_hist,
                     output_dir, img_base_fname):
    """
    Image processing pipeline entry point.
    :param img: image pixels array, in RGB color space
    :param output_dir: intermediate output directory
    :param img_base_fname: base filename for this image, None for disabling intermediate output
    :return: annotated output image
    """
    # Un-distort image
    img = cv2.undistort(img, mtx, dist, None, mtx)
    if img_base_fname is not None:
        output_img(img, os.path.join(output_dir, 'undistort', img_base_fname))

    # Parallelize the two pipelines
    threads = []
    thread_result = {}

    def call_vehicle_detection(t_result):
        t_result["vehicles"] = vehicle_detection_pipeline(img, vehicle_detection_params, vehicle_hist,
                                                          output_dir, img_base_fname)
    threads.append(threading.Thread(target=call_vehicle_detection, args=(thread_result,)))

    def call_land_finding(t_result):
        t_result["lanes"] = lane_finding_pipeline(img, lane_hist, output_dir, img_base_fname)
    threads.append(threading.Thread(target=call_land_finding, args=(thread_result,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    vehicles, searchbox, heatmap = thread_result["vehicles"]
    diag_window, lane_region, lane_pixels, radius, distance = thread_result["lanes"]

    result = np.copy(img)
    # Draw vehicle boxes
    for box in vehicles:
        cv2.rectangle(result, box[0], box[1], color=(0, 0, 255), thickness=6)
    # Overlay diagnosis windows for vehicle detection.
    insert_image(result, searchbox, x=10, y=30, shrinkage=2.6)
    insert_image(result, heatmap, x=520, y=30, shrinkage=3.5)
    # Overlay diagnosis windows for lane finding.
    insert_image(result, diag_window, x=900, y=30, shrinkage=3.5)
    # Highlight the lane region and pixels.
    result = cv2.addWeighted(result, 1, lane_region, 0.3, 0)
    result = cv2.addWeighted(result, .8, lane_pixels, 1, 0)
    # Add text for curvature and distance to lane center.
    cv2.putText(result, "Radius of Curvature: {:.1f}(m)".format(radius), org=(80, 630),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))
    cv2.putText(result, "Distance to Center : {:.3f}(m) {}".format(
        abs(distance), 'left' if distance < 0 else 'right'), org=(80, 680),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, required=False, default='../trained-model-params.p',
                        help='File path for the trained model and its parameters')
    parser.add_argument('--calibration-file', type=str, required=False, default='../calibration-params.p',
                        help='File path for camera calibration parameters')
    parser.add_argument('--image-dir', type=str, required=False, # default='../test_images',
                        help='Directory of images to process')
    parser.add_argument('--video-file', type=str, required=False, default='../test_video.mp4',
                        help="Video file to process")
    args = parser.parse_args()

    # Load camera calibration parameters.
    dist_pickle = pickle.load(open(args.calibration_file, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Load model and its parameters.
    dist_pickle = pickle.load(open(args.model_file, "rb"))
    vehicle_params = VehicleDetectionParams(classifier=dist_pickle["classifier"],
                                            feature_scaler=dist_pickle["feature_scaler"],
                                            feature_params=dist_pickle["feature_params"])
    print(vehicle_params.feature_params)

    if args.image_dir:
        images = glob.glob(os.path.join(args.image_dir, "*.jpg"))
        # images = ['../test_images/test1.jpg']
        for fname in sorted(images):
            print(fname)
            img = cv2.imread(fname)  # BGR
            out = process_pipeline(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mtx, dist, vehicle_params,
                                   None, None, '../output_images', os.path.basename(fname))
            output_img(out, os.path.join('../output_images', os.path.basename(fname)))

    if args.video_file:
        gen = fname_generator(max_num_frame=40)
        clip = VideoFileClip(args.video_file)  # .subclip(49, 51)
        lane_hist = LaneHistoryInfo()
        vehicle_hist = VehicleHistoryInfo()
        write_clip = clip.fl_image(lambda frame:  # RGB
            process_pipeline(frame, mtx, dist, vehicle_params, vehicle_hist, lane_hist,
                             '../output_images_' + os.path.basename(args.video_file), next(gen)))
        write_clip.write_videofile('../result_' + os.path.basename(args.video_file), audio=False)
