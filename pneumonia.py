import os 
import sys

import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob 
import scipy.signal as signal
import skimage.morphology as sm

# enter your Kaggle credentionals here
os.environ['KAGGLE_USERNAME']="lixiaoqi"
os.environ['KAGGLE_KEY']="a96b56107f574e1cf4e14e3822ca2c93"

os.chdir('E:\学习\ml-lessons')
# Root directory of the project
ROOT_DIR = os.path.abspath('./lesson3-data')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

#if not os.path.exists(ROOT_DIR):
#    os.makedirs(ROOT_DIR)
os.chdir(ROOT_DIR)####

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_test_images')

'''
- dicom_fps is a list of the dicom image path and filenames 
- image_annotions is a dictionary of the annotations keyed by the filenames
- parsing the dataset returns a list of the image filenames and the annotations dictionary
'''
# 将图片的绝对路径以list方式保存起来，这样就可以通过索引的方式直接获取图像路径及名称
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

#image_fps保存了list类型的图像路径及名称，image_annotations保存了每一张图像所对应的识别框的信息
def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 

# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

#  直方图均衡
def histep(img, nbr_bins=256):
    """Histogram equalization of a grayscale image."""
    # 获取直方图p(r)
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    
    # 获取T(r)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]
    
    #获取s, 并用s替换原始图像对应的灰度值
    result = np.interp(img.flatten(), bins[:-1], cdf)
    return result.reshape(img.shape), cdf

# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 
# 为在RSNA pneumonia数据集上训练模型而重写mask_rcnn中的Config类
class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    # 给该configuration取一个可识别的名字
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8 #!!!
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 64####256,64
    IMAGE_MAX_DIM = 64
    
    RPN_ANCHOR_SCALES = (32,64,128,256)####(32, 64, 128, 256)
    
    TRAIN_ROIS_PER_IMAGE = 16####32,16!!!
    
    MAX_GT_INSTANCES = 3
    
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.8####
    DETECTION_NMS_THRESHOLD = 0.1
    
    RPN_TRAIN_ANCHORS_PER_IMAGE = 16
    STEPS_PER_EPOCH = 100 ####300
    #TOP_DOWN_PYRAMID_SIZE = 32
    
config = DetectorConfig()
#config.display()

class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # 添加类
        self.add_class('pneumonia', 1, 'Lung Opacity')
   
        # 添加图像
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        
        # image = signal.medfilt(image, kernel_size=5)   # 对图片进行中值滤波
        # image = image.astype(np.uint8)
        # image = cv2.equalizeHist(image)  # 对图片进行直方图均衡处理
        image = sm.closing(image,sm.disk(5))  #用边长为5的圆形滤波器进行闭运算滤波
        
        image, _ = histep(image)   # 对图片进行直方图均衡处理
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

    
# 训练集
f=open(os.path.join(ROOT_DIR, 'stage_1_train_labels.csv'))
anns = pd.read_csv(f)
image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

# Modify this line to use more or fewer images for training/validation. 
# To use all images, do: image_fps_list = list(image_fps)
image_fps_list = list(image_fps)#[:1000] 
# 将数据分为训练集和验证集
# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
sorted(image_fps_list)
random.seed(42)
random.shuffle(image_fps_list)

validation_split = 0.1
split_index = int((1 - validation_split) * len(image_fps_list))
# 训练集和验证集
image_fps_train = image_fps_list[:split_index]
image_fps_val = image_fps_list[split_index:]

print('训练和测试样例个数:',len(image_fps_train), len(image_fps_val))

# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

# Image augmentation 
'''augmentation = iaa.SomeOf((0, 1), [
    
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #rotate=(-25, 25),
        shear=(-8, 8)
    ),
    iaa.Multiply((0.9, 1.1))
])'''
augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.OneOf([  # geometric transform
            iaa.Affine(
                scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
                translate_percent={"x": (-0.03, 0.03), "y": (-0.05, 0.05)},
                rotate=(-2, 2),
                shear=(-3, 3),
            ),
            iaa.PiecewiseAffine(scale=(0.002, 0.03)),
        ]),
        iaa.OneOf([  # brightness or contrast
            iaa.Multiply((0.85, 1.15)),
            iaa.ContrastNormalization((0.85, 1.15)),
        ]),
        iaa.OneOf([  # blur or sharpen
            iaa.GaussianBlur(sigma=(0.0, 0.12)),
            iaa.Sharpen(alpha=(0.0, 0.12)),
        ]),
])


# 加载预训练的COCO权重
COCO_MODEL_PATH = "./Mask_RCNN/samples/coco/mask_rcnn_coco.h5"

#设定权重的初始化方式，有imagenet，coco,last三种
init_with = "coco"
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last()[1], by_name=True)


NUM_EPOCHS = 37
L_R=0.005
# Train Mask-RCNN Model 
import warnings 
warnings.filterwarnings("ignore")
model.train(dataset_train, dataset_val, 
            learning_rate=0.02, 
            epochs=1, 
            layers='all',
            augmentation=None)
model.train(dataset_train, dataset_val, 
            learning_rate=L_R, 
            epochs=NUM_EPOCHS, 
            layers='all',
            augmentation=augmentation)
print('ok')

# select trained model 
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(model.model_dir))          
    
fps = []
# Pick last directory
for d in dir_names: 
    dir_name = os.path.join(model.model_dir, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else: 
      
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))

model_path = 'E:\学习\ml-lessons\lesson3-data\logs\pneumonia20181006T1038\mask_rcnn_pneumonia_0032.h5'
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)

# Make predictions on test images, write out sample submission 
def predict(image_fps, filepath='Mask_RCNN/sample_submission.csv', min_conf=0.95): 
    
    # assume square image
    
    with open(filepath, 'w') as file:
        out_str = "patientId,PredictionString"
        file.write(out_str+"\n")
        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array
            
            # image = signal.medfilt(image, kernel_size=5)   # 对图片进行中值滤波
            # image = image.astype(np.uint8)
            # image = cv2.equalizeHist(image)  # 对图片进行直方图均衡处理
            image = sm.closing(image,sm.disk(5))  #用边长为5的圆形滤波器进行闭运算滤波
            image, _ = histep(image)   # 对图片进行直方图均衡处理
            
            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1) 
            
            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            out_str = ""
            out_str += patient_id 
            out_str += ","
            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0: 
                pass
            else: 
                num_instances = len(r['rois'])
                for i in range(num_instances): 
                    if r['scores'][i] > min_conf: 
                        out_str += ' '
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height 
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1 
                        height = r['rois'][i][2] - y1 
                        bboxes_str = "{} {} {} {}".format(x1, y1, width, height)    
                        out_str += bboxes_str

            file.write(out_str+"\n")

sample_submission_fp = 'Mask_RCNN/sample_submission.csv'
predict(test_image_fps, filepath=sample_submission_fp)