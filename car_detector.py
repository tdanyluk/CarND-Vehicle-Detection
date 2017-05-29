import time
import pickle
import glob
from collections import deque
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

import lesson_functions


class CarDetector:
    def __init__(self):
        self.random_seed = 19920828

        self.color_space = "YCrCb"  # Color space
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 32  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.cell_per_step = 2  # Number of cells that the sliding window steps in each step
        self.lookback_frames = 10
        self.heat_threshold = 10

        self.cars_path_glob = "vehicles/*/*.png"
        self.non_cars_path_glob = "non-vehicles/*/*.png"

        self.features_pickle_name = "features.pickle"
        self.svc_pickle_name = "svc.pickle"
        self.scaler_pickle_name = "scaler.pickle"

        self.image_size = (1280, 720)
        self.image_shape = (self.image_size[1], self.image_size[0])
        self.image_shape_3 = (self.image_size[1], self.image_size[0], 3)

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.scaler = None
        self.svc = None

        self.heat_image = np.zeros(self.image_shape, dtype=np.uint16)
        self.active_rect_queue = deque([])  # rect: ((x1,y1), (x2, y2))

    def _extract_features(self, file_names):
        return lesson_functions.extract_features(file_names, color_space=self.color_space,
                                                 spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                 orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                 cell_per_block=self.cell_per_block,
                                                 hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                                 hist_feat=self.hist_feat, hog_feat=self.hog_feat)

    def extract_training_features(self):
        t0 = time.time()

        cars = glob.glob(self.cars_path_glob)
        notcars = glob.glob(self.non_cars_path_glob)

        car = lesson_functions.read_image(random.choice(cars))
        not_car = lesson_functions.read_image(random.choice(notcars))
        self.visualise_car_not_car(car, not_car)
        self.visualise_hog(car, not_car)

        car_features = self._extract_features(cars)
        notcar_features = self._extract_features(notcars)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        y = np.hstack((np.ones(len(car_features)),
                       np.zeros(len(notcar_features))))

        self.scaler = StandardScaler().fit(X)
        scaled_X = self.scaler.transform(X)

        X = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=self.random_seed)

        scaled_X = None
        y = None

        pickle.dump(self.scaler, open(self.scaler_pickle_name, "wb"))
        pickle.dump((self.X_train, self.X_test, self.y_train, self.y_test),
                    open(self.features_pickle_name, "wb"))

        t1 = time.time()
        print('extract_training_features: ', round(t1 - t0, 2), 's')

    def load_training_features(self):
        self.X_train, self.X_test, self.y_train, self.y_test = pickle.load(
            open(self.features_pickle_name, "rb"))

    def fit_classifier(self):
        self.svc = LinearSVC()
        t0 = time.time()
        self.svc.fit(self.X_train, self.y_train)
        t1 = time.time()
        print(round(t1 - t0, 2), 'Seconds to train SVC...')
        print('Test Accuracy of SVC = ', round(
            self.svc.score(self.X_test, self.y_test), 4))
        pickle.dump(self.svc, open(self.svc_pickle_name, "wb"))

    def load_classifier(self):
        self.scaler = pickle.load(open(self.scaler_pickle_name, "rb"))
        self.svc = pickle.load(open(self.svc_pickle_name, "rb"))

    def _find_cars(self, img, ystart, ystop, scale):
        return lesson_functions.find_cars(img, ystart, ystop, scale, self.svc, self.scaler, self.orient,
                                          self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins, self.cell_per_step,
                                          self.color_space)

    def calculate_car_rectangles(self, image):
        rects = []
        rects += self._find_cars(image, 380, 500, 1.0)
        rects += self._find_cars(image, 380, 670, 1.5)
        rects += self._find_cars(image, 380, 670, 2.0)
        return rects

    def draw_rectangles(self, image, rects, color=(0, 0, 1), thick=6):
        return lesson_functions.draw_boxes(image, rects, color, thick)

    def add_heat(self, rectangles):
        if len(self.active_rect_queue) >= self.lookback_frames:
            rects_to_remove = self.active_rect_queue.popleft()
            for rect in rects_to_remove:
                self.heat_image[rect[0][1]:rect[1][1],
                                rect[0][0]:rect[1][0]] -= 1

        for rect in rectangles:
            self.heat_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] += 1

        self.active_rect_queue.append(rectangles)

    def create_heat_image(self, rectangles):
        heat_image = np.zeros(self.image_shape, dtype=np.uint16)
        for rect in rectangles:
            heat_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] += 1
        return heat_image

    def _get_thresholded_heat_image(self):
        heatmap = np.copy(self.heat_image)
        heatmap[heatmap <= self.heat_threshold] = 0
        return heatmap

    def calculate_robust_car_rectangles(self, image):
        heatmap = self._get_thresholded_heat_image()
        labels = label(heatmap)
        return lesson_functions.draw_labeled_bboxes(np.copy(image), labels)

    def process_image(self, file_name):
        img = lesson_functions.read_image(file_name)
        rects = self.calculate_car_rectangles(img)
        return self.draw_rectangles(img, rects, color=(0, 0, 1))

    def _heat_image_to_rgb(self, heat_image):
        cmap = plt.cm.hot
        norm = plt.Normalize(vmin=heat_image.min(), vmax=heat_image.max())
        return cmap(norm(heat_image))[:, :, 0:3].astype(np.float32)

    def _process_frame(self, frame):
        frame = frame.astype(np.float32) / 255

        rects = self.calculate_car_rectangles(frame)
        self.add_heat(rects)
        #rgb_heat = self._heat_image_to_rgb(self.heat_image)
        cars = self.calculate_robust_car_rectangles(frame)
        #cars = self.draw_rectangles(cars, rects, color=(0, 1, 0), thick=1)

        #frame = cv2.addWeighted(cars, 0.6, rgb_heat, 0.4, 0)
        frame = cars
        return (frame * 255).astype(np.uint8)

    def process_video(self, in_file_name, out_file_name):
        self.heat_image = np.zeros(self.image_shape, dtype=np.uint16)
        self.active_rect_queue = deque([])

        src_clip = VideoFileClip(in_file_name)
        dst_clip = src_clip.fl_image(lambda frame: self._process_frame(frame))
        dst_clip.write_videofile(out_file_name, audio=False)

    def visualise(self, images, titles=None, cols=2, cell_width=6, target=None, cmap='gray'):
        rows = (len(images) + cols - 1) // cols
        cell_height = int(
            cell_width * images[0].shape[0] / images[0].shape[1] * 1.5)
        plt.figure(figsize=(cols * cell_width, rows * cell_height))
        for i, img in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            img_dims = len(img.shape)
            if img_dims < 3:
                plt.imshow(img, cmap=cmap)
            else:
                plt.imshow(img)
            if titles is not None:
                plt.title(titles[i])
            else:
                plt.title('Image {0}'.format(i + 1))
        if target is None:
            plt.show()
        else:
            plt.savefig(target)

    def visualise_car_not_car(self, car, not_car):
        self.visualise([car, not_car], ["Car", "Not car"],
                       cell_width=2, target="writeup_images/car_not_car.png")

    def get_hog_image(self, img):
        _, img = lesson_functions.get_hog_features(
            img, self.orient, self.pix_per_cell, self.cell_per_block, vis=True)
        return img

    def visualise_hog(self, car, not_car):
        car = lesson_functions.convert_color(car, self.color_space)
        not_car = lesson_functions.convert_color(not_car, self.color_space)

        images = []
        titles = []
        for channel in range(0, 3):
            images.extend([car[:, :, channel],
                           self.get_hog_image(car[:, :, channel]),
                           not_car[:, :, channel],
                           self.get_hog_image(not_car[:, :, channel])])
            titles.extend(["Car CH-{0}".format(channel + 1),
                           "Car CH-{0} Hog".format(channel + 1),
                           "Not Car CH-{0}".format(channel + 1),
                           "Not Car CH-{0} Hog".format(channel + 1)])

        self.visualise(images, titles, cols=4, cell_width=4,
                       target="output_images/hog.png")

    def visualise_sliding_window(self, path_glob):
        test_images = glob.glob(path_glob)
        processed = []
        for img in test_images:
            processed.append(self.process_image(img))
        self.visualise(processed, target="output_images/sliding_window.png")

    def visualise_heatmaps(self, path_glob):
        test_images = glob.glob(path_glob)
        images = []
        combined_heat_map = np.zeros_like(self.heat_image)

        last_image = None

        for img in test_images:
            img = lesson_functions.read_image(img)
            last_image = img
            rects = self.calculate_car_rectangles(img)

            images.append(self.draw_rectangles(img, rects))
            heat_image = self.create_heat_image(rects)
            combined_heat_map += heat_image
            images.append(heat_image)

        car_detector.visualise(images, cell_width=4,
                               cmap='hot', target="output_images/heat.png")
        combined_heat_map[combined_heat_map <= self.heat_threshold] = 0
        labels = label(combined_heat_map)
        bounds = lesson_functions.draw_labeled_bboxes(last_image, labels)
        car_detector.visualise([labels[0]], [""], cell_width=6,
                               cols=1, cmap='gray', target="output_images/labeled.png")
        car_detector.visualise([bounds], [""], cell_width=6,
                               cols=1, cmap='gray', target="output_images/bounds.png")


car_detector = CarDetector()

# Uncomment these lines to regenerate training features and refit the model
#car_detector.extract_training_features()
#car_detector.load_training_features()
#car_detector.fit_classifier()
car_detector.load_classifier()

car_detector.visualise_sliding_window("test_images/for_sliding_window/*.png")
car_detector.visualise_heatmaps("test_images/sequence/*.png")

#car_detector.process_video("test_video.mp4", "test_video_out.mp4")
car_detector.process_video("project_video.mp4", "project_video_out.mp4")

# Uncomment for profiling
#import cProfile
#cProfile.run('car_detector.process_video("test_video.mp4", "test_video_out.mp4")')
