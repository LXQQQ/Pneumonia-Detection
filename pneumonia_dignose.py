# -*- coding: utf-8 -*-

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


# Root directory of the project
ROOT_DIR = os.path.abspath("./data")

#Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'weights')

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)


Mask_RCNN_DIR = os.path.abspath("../Mask_RCNN")
sys.path.append(Mask_RCNN_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
train_label_dir = os.path.join(ROOT_DIR, 'stage_1_train_labels.csv')
test_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_test_images')


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


# 为在pneumonia数据集上训练模型而重写mask_rcnn中的Config类
class DetectorConfig(Config):
    # 给该configuration取一个可识别的名字
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2   # background + 1 pneumonia classes
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    # RPN_ANCHOR_RATIOS = [0.3, 0.5, 1, 2]
    
    TRAIN_ROIS_PER_IMAGE = 32
    
    MAX_GT_INSTANCES = 4 
    
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.01
    
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 16 
    # TOP_DOWN_PYRAMID_SIZE = 32
    STEPS_PER_EPOCH = 400

config = DetectorConfig()
config.display()


class DetectorDataset(utils.Dataset):
    #Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # 添加类
        self.add_class('pneumonia', 1, 'Lung Opacity')
        
        # 添加图像
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        """Return the path of the image."""
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
        image, _ = histep(image)   # 对图片进行直方图均衡处理
        # If grayscale, convert to RGB for consistency
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



# 数据集
anns = pd.read_csv(train_label_dir)
image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

# 将数据分为训练集和验证集
image_fps_list = list(image_fps)
random.seed(42)
random.shuffle(image_fps_list)

# 训练集和验证集
val_size = 1500
image_fps_val = image_fps_list[:val_size]
image_fps_train = image_fps_list[val_size:]

print(len(image_fps_train), len(image_fps_val))

# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# Show annotation(s) for a DICOM image 
# choice() 方法返回一个列表，元组或字符串的随机项。
# test_fp = random.choice(image_fps_train)
# print(image_annotations[test_fp])

# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()




# 模型
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

# Image augmentation (light but constant)
augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),  # 水平翻转
        # geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.Multiply((0.97, 1.03))
])

# model.load_weights("./data/weights/pneumonia20181015T2020/mask_rcnn_pneumonia_0035.h5", by_name=True)

# 加载预训练的COCO权重
COCO_MODEL_PATH = "../Mask_RCNN/samples/coco/mask_rcnn_coco.h5"

# 加载预训练的IMAGENET权重
IMAGENET_MODEL_PATH = "../Mask_RCNN/samples/ImageNet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

#设定权重的初始化方式，有imagenet，coco,last三种
init_with = "coco"
if init_with == "imagenet":
    model.load_weights(IMAGENET_MODEL_PATH, by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last()[1], by_name=True)

NUM_EPOCHS = 40
LEARNING_RATE = 0.005
# Train Mask-RCNN Model 
import warnings 
warnings.filterwarnings("ignore")
# 头两个epoch使用较大的learning rate，并且只训练heads部分
model.train(dataset_train, dataset_val, 
            learning_rate=LEARNING_RATE*2, 
            epochs=2, 
            layers='heads',
            augmentation=None)  # no need to augment yet
model.train(dataset_train, dataset_val, 
            learning_rate=LEARNING_RATE, 
            epochs=6, 
            layers='all',
            augmentation=augmentation)
# 后面的epoch使用更小的learning rate
model.train(dataset_train, dataset_val, 
            learning_rate=LEARNING_RATE/5, 
            epochs=16, 
            layers='all',
            augmentation=augmentation)
model.train(dataset_train, dataset_val, 
            learning_rate=LEARNING_RATE/50, 
            epochs=30, 
            layers='all',
            augmentation=augmentation)
model.train(dataset_train, dataset_val, 
            learning_rate=LEARNING_RATE/500, 
            epochs=NUM_EPOCHS, 
            layers='all',
            augmentation=augmentation)


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

# sorted(A)[-1]是一个有序序列A的最后一个元素
model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)

model.load_weights("./data/weights/pneumonia20181018T1445/mask_rcnn_pneumonia_0033.h5", by_name=True)

'''
#load traind weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
'''


# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


# 这里显示几个预测结果与真实情况对比的例子
dataset = dataset_val
fig = plt.figure(figsize=(10, 30))

for i in range(6):
    image_id = random.choice(dataset.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
        
    plt.subplot(6, 2, 2*i + 1)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
        
    plt.subplot(6, 2, 2*i + 2)
    results = model.detect([original_image]) #, verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])


# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)

def iou(box1, box2):
    y11, x11, y12, x12 = box1
    y21, x21, y22, x22 = box2
    
    w1 = x12 - x11
    h1 = y12 - y11
    w2 = x22 - x21
    h2 = y22 - y21
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    
    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    new_box = [yi1, xi1, yi2, xi2]
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0, new_box
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union, new_box

def choose_intersect(model, config, image, shreshhold=0.2):
    result1 = model.detect([image])
    r1 = result1[0]
    assert( len(r1['rois']) == len(r1['class_ids']) == len(r1['scores']) )
    
    temp_img = np.fliplr(image)
    result2 = model.detect([temp_img])
    r2 = result2[0]
    assert( len(r2['rois']) == len(r2['class_ids']) == len(r2['scores']) )
    
    #r2['rois'] = np.fliplr(r2['rois'])
    num = len(r2['rois'])
    for k in range(num):
        w = r2['rois'][k][3] - r2['rois'][k][1]
        r2['rois'][k][1] = config.IMAGE_MIN_DIM - r2['rois'][k][1] - w
        r2['rois'][k][3] = r2['rois'][k][1] + w
    
    for i, bt in enumerate(r1['rois']):
        max_intersect = 0
        for j, bp in enumerate(r2['rois']):
            intersect, new_box = iou(bt, bp)
            if intersect >= shreshhold and intersect > max_intersect:
                max_intersect = intersect
                r1['rois'][i] = new_box
                r1['scores'][i] = (r1['scores'][i] + r2['scores'][j]) / 2.0
        if max_intersect == 0:
            r1['scores'][i] = 0
    
    return r1

# Make predictions on test images, write out sample submission 
def predict(image_fps, filepath='submission.csv', min_conf=0.95): 
    
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    with open(filepath, 'w') as file:
      file.write("patientId,PredictionString\n")
      
      for image_id in tqdm(image_fps): 
        ds = pydicom.read_file(image_id)
        image = ds.pixel_array
        # image = signal.medfilt(image, kernel_size=5)   # 对图片进行中值滤波
        # image = image.astype(np.uint8)
        # image = cv2.equalizeHist(image)  # 对图片进行直方图均衡处理
        image, _ = histep(image)   # 对图片进行直方图均衡处理
          
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1) 
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
            
        patient_id = os.path.splitext(os.path.basename(image_id))[0]

        '''
        results = model.detect([image])
        r = results[0]
        '''

        r = choose_intersect(model, config, image, shreshhold=0.2)

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
                    bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
                                                      width*resize_factor, height*resize_factor)    
                    out_str += bboxes_str

        file.write(out_str+"\n")


# predict
sample_submission_fp = 'submission.csv'
predict(test_image_fps, filepath=sample_submission_fp)
