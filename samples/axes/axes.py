"""
Mask R-CNN
Train on the axes dataset and implement color splash effect.
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 axes.py train --dataset="../../dataset/axes_clear" --weights=coco

    # Resume training a model that you had trained earlier
    python3 axes.py train --dataset=/path/to/axis/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 axes.py train --dataset=/path/to/axes/dataset --weights=imagenet

    # Apply color splash to an image
    python3 axes.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 axes.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import time
import numpy as np
import numpy as np
import tensorflow as tf
import cv2
import cv2 as cv
import skimage.draw
import imgaug
from skimage import draw, morphology
import glob
from xml.dom.minidom import parse, parseString
import warnings
warnings.filterwarnings("ignore")

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")


from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DELTA = 5
EPOCHS = 80

############################################################
#  Configurations
############################################################


class AxisConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "axis"

    USE_MINI_MASK = False
    LEARNING_RATE = 0.005


    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2048

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + axis
    VALIDATION_STEPS = 10

    STEPS_PER_EPOCH = 998

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BACKBONE = "resnet101"

    IMAGE_RESIZE_MODE = "crop"



############################################################
#  Dataset
############################################################

class AxisDataset(utils.Dataset):
    def extract_info_from_json(self, path_to_json, path_to_jpg):
        jpg_filename = path_to_jpg.split(os.sep)[-1]
        # print(path_to_jpg)
        img = cv2.imread(path_to_jpg)
        
        f = open(path_to_json, 'r', encoding='utf-8')
        data = json.load(f)
        # Initialise the info dict
        info_dict = {}
        info_dict['bboxes'] = []
        info_dict['filename'] = jpg_filename
        # print(path_to_jpg)
        info_dict['image_size'] = (img.shape[1], img.shape[0], img.shape[2])
        layers_list = data['layers']
        for layer in layers_list:
            if not layer.get('name'):
                logging.info('Layer doesnt have name key!    ' + json_path)
                continue
            if layer['name'] == 'Оси':
                if len(layer['shapes']) < 1:
                    logging.info('Axis shapes is empty!    ' + json_path)
                    continue
                for shape in layer['shapes']:
                    bbox = {}
                    if (
                        shape['name'].find("ыше")!=-1 or
                        shape['name'].find("иже")!=-1 or
                        shape['name'].find("евее")!=-1 or
                        shape['name'].find("равее")!=-1
                    ):
                        continue
                    bbox["class"] = 'axis_line'
                    values_dict = {}
                    target_str = shape['svg']
                    for value in ['x1', 'y1', 'x2', 'y2']:
                        for split_part in target_str.split(' '):
                            if split_part.startswith(value):
                                values_dict[value] = int(float(split_part.split('"')[1]))
                    bbox["x1"] = min(values_dict['x1'], info_dict['image_size'][0] - 2)
                    bbox["y1"] = min(values_dict['y1'], info_dict['image_size'][1] - 2)
                    bbox["x2"] = min(values_dict['x2'], info_dict['image_size'][0] - 2)
                    bbox["y2"] = min(values_dict['y2'], info_dict['image_size'][1] - 2)
                    info_dict['bboxes'].append(bbox)
        return info_dict

    def load_axis(self, dataset_dir, subset):
        """Load a subset of the Axis dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("axis", 1, "axis")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        jsons_dataset_dir = os.path.join(dataset_dir, subset, 'jsons/')
        # print(jsons_dataset_dir)
        for json_path in glob.glob(jsons_dataset_dir + '*.json'):
            try:
                jpg_path = json_path.replace('jsons', 'jpgs').replace('.json', '.jpg')

                info_dict = self.extract_info_from_json(json_path, jpg_path)

                image_path = jpg_path
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "axis",
                    image_id=info_dict['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    bboxes=info_dict['bboxes'])
            except:
                print(json_path, jpg_path)
                continue

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a axis dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "axis":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([len(info["bboxes"]), info["height"], info["width"]],
                        dtype=np.uint8)
        # mask = None

        # start = time.time()
        # sum_replace = 0

        for i, bbox in enumerate(info["bboxes"]):
            # rr, cc = skimage.draw.line(bbox['y1'],bbox['x1'] ,bbox['y2'] ,bbox['x2'])
            # print(bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])

            mask_layer = np.zeros([info["height"], info["width"]], dtype=np.uint8)
            mask_layer = cv.line(mask_layer, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), 1, 20)
            # mask[rr, cc, i] = 1
            # mask[:,:,i] = morphology.dilation(mask[:,:,i], morphology.disk(radius=10))
            # each_start = time.time()
            mask[i] = mask_layer
            # each_time = time.time() - each_start
            # sum_replace += each_time
            # if mask is None:
            #     mask = mask_layer[:,:, None]
            # else:
            #     mask = np.concatenate((mask, mask_layer[:,:,None]), axis=2)
        # start_transpose = time.time()
        mask = np.transpose(mask, (1, 2, 0))
        # end_transpose = time.time() - start_transpose
        # print(mask_layer.shape)
        # print(mask.shape)
        # print("Time_to_image:", time.time() - start)
        # print("Time_to_transpose:", end_transpose)
        # print("Time_to_replace:", sum_replace)
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "axis":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect axis.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/axis/dataset/",
                        help='Directory of the Axis dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AxisConfig()
    else:
        class InferenceConfig(AxisConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        print(model.find_last())
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Sequential([imgaug.augmenters.Fliplr(0.5),
                                                    imgaug.augmenters.Flipud(0.5),
                                                     imgaug.augmenters.Affine(
                                                                             rotate=(-360, 360),
                                                                             ),
                                                    # imgaug.augmenters.AddToHueAndSaturation((-50, 50), per_channel=True)
                                                     ])

        # *** This training schedule is an example. Update to your needs ***

        # Training dataset.
        dataset_train = AxisDataset()
        dataset_train.load_axis(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = AxisDataset()
        dataset_val.load_axis(args.dataset, "val")
        dataset_val.prepare()

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=EPOCHS,
                    layers='heads',
                    augmentation=augmentation)

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=60,
        #             layers='4+',
        #             augmentation=augmentation)
        #
        #  # Training - Stage 3
        #  # Fine tune all layers
        # print("Fine tune all layers")
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=80,
        #             layers='all',
        #             augmentation=augmentation)

    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
