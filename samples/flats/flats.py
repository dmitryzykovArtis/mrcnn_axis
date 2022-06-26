"""
Mask R-CNN
Train on the flats dataset and implement color splash effect.
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python flats.py train --dataset="../../../../datasets/union_val" --weights=coco

    # Resume training a model that you had trained earlier
    python3 flats.py train --dataset=/path/to/flat/datasets --weights=last

    # Train a new model starting from ImageNet weights
    python3 flats.py train --dataset=/path/to/flats/datasets --weights=imagenet

    # Apply color splash to an image
    python3 flats.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 flats.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import imgaug
import time
import numpy as np
import numpy as np
import cv2 as cv
import skimage.draw
from skimage import draw, morphology
import glob
import cv2
from xml.dom.minidom import parse, parseString

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")


from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
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
EPOCHS = 40

############################################################
#  Configurations
############################################################


class FlatConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "flat"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + flat

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 368

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BACKBONE = "resnet50"

    USE_MINI_MASK = False

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 2.
    }



############################################################
#  Dataset
############################################################

class FlatDataset(utils.Dataset):
    def extract_info_from_json(self, path_to_json, path_to_jpg):
        jpg_filename = path_to_jpg.split(os.sep)[-1]
        # print(path_to_jpg)
        img = cv2.imread(path_to_jpg)
        
        f = open(path_to_json, 'r', encoding='utf-8')
        data = json.load(f)
        # Initialise the info dict
        info_dict = {}
        info_dict['polylines'] = []
        info_dict['filename'] = jpg_filename
        # print(path_to_jpg)
        info_dict['image_size'] = (img.shape[1], img.shape[0], img.shape[2])
        layers_list = data['layers']
        var_start = ['width',
                     'height',
                     'x',
                     'y']
        for layer in layers_list:
            if not layer.get('name'):
                logging.info('Layer doesnt have name key!    ' + json_path)
                continue
            if layer['name'] == 'Пользовательские фигуры':
                if len(layer['shapes']) < 1:
                    logging.info('Flat shapes is empty!    ' + json_path)
                    continue
                for shape in layer['shapes']:
                    poly = {}
                    class_type = 1
                    if "spaceType" in shape and shape['spaceType'] == 'common-areas':
                        class_type = 2
                    poly["class"] = 'flat_poly'
                    if shape['type'] == 'ellipse':
                        continue
                    if shape['type'] == 'rect':
                        variable_dict = {}
                        for variable in shape['svg'].split(' '):
                            for start in var_start:
                                if variable.startswith(start):
                                    variable.split('"')[1]
                                    variable_dict[start] = int(float(variable.split('"')[1]))
                        width = variable_dict['width']
                        height = variable_dict['height']
                        x = variable_dict['x']
                        y = variable_dict['y']
                        poly = [
                            [x, y],
                            [x + width, y],
                            [x + width, y + height],
                            [x, y + height]
                        ]
                    else:
                        points = shape['svg'].split('"')[1].split(' ')
                        poly = [[float(string.split(',')[0]), float(string.split(',')[1])] for string in points]
                    poly_and_class =[np.array(poly, np.int32), class_type]
                    info_dict['polylines'].append(poly_and_class)
        return info_dict

    def load_flat(self, dataset_dir, subset):
        """Load a subset of the Flat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val

        ТУТ все ОК
        """
        # Add classes. We have only one class to add.
        self.add_class("flat", 1, "flat")
        self.add_class("common_area", 2, "common_area")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        jsons_dataset_dir = os.path.join(dataset_dir, subset, 'jsons/')
        # print(jsons_dataset_dir)
        for json_path in glob.glob(jsons_dataset_dir + '*.json'):
            jpg_path = json_path.replace('jsons', 'jpgs').replace('.json', '.jpg')
            info_dict = self.extract_info_from_json(json_path, jpg_path)

            image_path = jpg_path
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "flat",
                image_id=info_dict['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polylines=info_dict['polylines'])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a flat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "flat":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([len(info["polylines"]), info["height"], info["width"]],
                        dtype=np.uint8)
        class_ids = []
        # mask = None

        # start = time.time()
        # sum_replace = 0
        for i, poly_and_class in enumerate(info["polylines"]):
            poly, class_type = poly_and_class
            mask_layer = np.zeros([info["height"], info["width"]], dtype=np.uint8)
            # print(poly)
            mask_layer = cv2.fillPoly(mask_layer, [poly], 1)
            mask[i] = mask_layer
            class_ids.append(class_type)
        class_ids = np.array(class_ids, dtype=np.int32)
        mask = np.transpose(mask, (1, 2, 0))
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "flat":
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
        description='Train Mask R-CNN to detect flat.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/flat/dataset/",
                        help='Directory of the Flat dataset')
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
        config = FlatConfig()
    else:
        class InferenceConfig(FlatConfig):
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
        augmentation = imgaug.augmenters.Sequential([
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.Flipud(0.5),
            # imgaug.augmenters.Rot90((1, 3), keep_size=False),
                                                     ])

        # *** This training schedule is an example. Update to your needs ***

        dataset_train = FlatDataset()
        dataset_train.load_flat(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = FlatDataset()
        dataset_val.load_flat(args.dataset, "val")
        dataset_val.prepare()

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=EPOCHS,
                    layers='heads',
                    augmentation=augmentation)

        #Training - Stage 2
        #Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=80,
                    layers='4+',
                    augmentation=augmentation)

        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=100,
                    layers='4+',
                    augmentation=augmentation)
        #
        # Training - Stage 3
        # Fine tune all layers
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
