"""

Model utils

"""
from datetime import datetime
import logging
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from utils.cv_utils import convert_from_image_to_cv2, convert_from_cv2_to_image
from utils.page import createBbox
from shapely import geometry
import torch
try:
    import segmentation_models_pytorch
except:
    logging.info(f"Unet not supported")
from torchvision import transforms as T
try:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.utils.visualizer import ColorMode
except:
    logging.info(f"Detectron2 not supported")
    
import pickle

try:
    from ultralytics import YOLO
except:
    logging.info(f"YOLO not supported")

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from torch import nn
try:
    from kraken import blla
    from kraken.lib import vgsl
except:
    logging.info(f"Kraken not supported")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_inference_cfg(opts, config_file_path, checkpoint_url, num_classes, device):
    cfg = get_cfg()
    if config_file_path.startswith("/"):
        cfg.merge_from_file(config_file_path)
    else:
        cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    if checkpoint_url.startswith("/"):
        cfg.MODEL.WEIGHTS = checkpoint_url
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 6000
    cfg.SOLVER.STEPS = []

    #    cfg.TEST.EVAL_PERIOD = 100
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = opts.device
    # for line detection
    if opts.small_objects == True:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0, 3.0, 4.0]]
        cfg.MODEL.BACKBONE.FREEZE_AT = 2  # Reducing stride for more granularity

        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14  # Increase from 7x7 or 14x14
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3  # Lower the IoU threshold
    return cfg
    
def load_text_recognize_model(opts, device):
    if opts.text_recognize_engine == "trocr":
        opts.text_recognize_processor = TrOCRProcessor.from_pretrained(opts.text_recognize_processor_path)
        opts.text_recognize_model = VisionEncoderDecoderModel.from_pretrained(opts.text_recognize_model_path).to(device).half()
        for param in opts.text_recognize_model.parameters():
            param.requires_grad = False
        opts.text_recognize_model.eval()
        if opts.cell_classifier_path != "":
            opts.cell_classifier = YOLO(opts.cell_classifier_path).to(opts.device)
            
def load_cell_classifier(opts, device):
   if opts.cell_classifier_path != "":
       opts.cell_classifier = YOLO(opts.cell_classifier_path).to(opts.device)
       

def load_region_model(opts, device):
    if opts.region_ensemble:
            region_predictor  = YOLO(opts.ensemble_model_path1)
            region_predictor.to(opts.device)
            region_predictor2 = SegformerForSemanticSegmentation.from_pretrained(opts.ensemble_model_path2)
            region_predictor2.to(opts.device)
            
    else:
        if opts.region_detection_model.lower() == "maskr-cnn" or opts.region_detection_model.lower() == "maskrcnn":
            region_cfg = get_inference_cfg(opts, opts.region_config_path, opts.region_model_path,
                                       opts.region_num_classes, opts.device)

            region_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            region_predictor = DefaultPredictor(region_cfg)
        elif opts.region_detection_model.lower() == "unet":
            region_predictor = torch.load(opts.region_model_path, map_location=torch.device(opts.device))

        elif opts.region_detection_model.lower() == "unetplusplus":
            region_predictor = torch.load(opts.region_model_path, map_location=torch.device(opts.device))

        elif opts.region_detection_model.lower() == "yolo":
            region_predictor = YOLO(opts.region_model_path)
            region_predictor.to(opts.device)

        elif opts.region_detection_model.lower() == "segformer":
            region_predictor = SegformerForSemanticSegmentation.from_pretrained(opts.region_model_path)
            region_predictor.to(opts.device)
        elif opts.region_detection_model.lower() == "transformer":
            region_predictor = torch.load(opts.region_model_path)
            region_predictor.to(opts.device)

    if opts.region_ensemble:
        return region_predictor, region_predictor2
    else:
        return region_predictor
def load_table_models(opts, device):
    if opts.table_detection_model.lower() == "yolo":
            table_predictor = YOLO(opts.table_detection_model_path)
            table_predictor.to(opts.device)
            table_structure_predictor = YOLO(opts.table_structure_detection_model_path)
            table_predictor.to(opts.device)
    return table_predictor, table_structure_predictor

def load_line_model(opts, device):
    if opts.line_detection_model.lower() == "maskrcnn" or opts.line_detection_model.lower() == "maskr-cnn":
        line_cfg = get_inference_cfg(opts, opts.line_config_path, opts.line_model_path,  len(opts.line_classes),
                                     opts.device)
        line_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        line_predictor = DefaultPredictor(line_cfg)
    elif opts.line_detection_model.lower() == "yolo":
        line_predictor = YOLO(opts.line_model_path)
        line_predictor.to(opts.device)
    elif opts.line_detection_model.lower() == "kraken":
        if not opts.package_kraken:
            raise ModuleNotFoundError()
        line_predictor = vgsl.TorchVGSLModel.load_model(opts.kraken_model_path)
    elif opts.line_detection_model.lower() == "segformer":
        line_predictor = SegformerForSemanticSegmentation.from_pretrained(opts.line_model_path)
        line_predictor.to(opts.device)
    return line_predictor
import cv2
import numpy as np
import matplotlib.pyplot as plt 

def only_area_of_image(image, polygon):
    if isinstance(image, str):
        image = cv2.imread(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv2.boundingRect(polygon)
    cropped_image = masked_image[y:y+h, x:x+w]
    return cropped_image

cell_types = ['empty', 'line', 'multi-line', 'same-as']

def classify_images(opts, images, line_indices, classifier):

    nonempty_images = []
    nonempty_indices = []
    nonempty_loc = []
    same_as = []
    results = classifier.predict(images, batch=16, verbose=False, device=opts.device)
    pred_labels = []
    for idx, (r, c_image, c_idx) in enumerate(zip(results, images, line_indices)):
        max_index = torch.argmax(r.probs.data).item()
        label = r.names[max_index]
        if label in ['line', 'multi-line']:
            nonempty_images.append(c_image)
            nonempty_indices.append(idx)
            nonempty_loc.append(c_idx)
        elif label in ['same-as']:
            same_as.append(idx)
    return nonempty_images, nonempty_indices, nonempty_loc, same_as



def classify_cell_type_in_batch(opts, images, classifier):

    results = classifier.predict(images, batch=16, verbose=False, device=opts.device)
    pred_labels = []
    for r in results:
        max_index = torch.argmax(r.probs.data).item()
        label = r.names[max_index]
        pred_labels.append(label)
    return pred_labels
    
def classify_cell_type(opts, image, classifier, area):
    if isinstance(area, np.ndarray):
        c_image = only_area_of_image(image, area)
        results = classifier.predict(c_image, verbose=False, device=opts.device)
        max_index = torch.argmax(results[0].probs.data).item()
    return results[0].names[max_index]

def process_batches(images, batch_size):

    for i in range(0, len(images), batch_size):
        yield images[i:i+batch_size]
        

def predict_image(opts, model, imagepath, model_type=None, area=None):
    image = cv2.imread(imagepath)
    height, width, channels = image.shape
    if isinstance(area, np.ndarray):
        image = only_area_of_image(image, area)
    if model_type == "line":
        if opts.line_detection_model.lower() == "maskrcnn":
            outputs = model(image)
        elif opts.line_detection_model.lower() == "yolo":
            imgsz = (opts.yolo_image_size,opts.yolo_image_size)
            resized_img = cv2.resize(image, imgsz)
            results = model.predict(resized_img, verbose=False, device=opts.device)
            outputs = results[0]
        elif opts.line_detection_model.lower() == "kraken":
#            image = Image.open(imagepath).convert("RGB")
            image2 = convert_from_cv2_to_image(image).convert("RGB")
            outputs = blla.segment(image2, model=model)
        elif opts.line_detection_model.lower() == "segformer":
            feature_extractor_inference = SegformerImageProcessor(do_random_crop=False, do_pad=False)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(opts.device)
            model.eval()
            outputs = model(pixel_values=pixel_values)

    elif model_type == "region":
        if opts.region_detection_model.lower() == "maskrcnn":
            outputs = model(image)
        elif opts.region_detection_model.lower() == "unet" or opts.region_detection_model.lower() == "unetplusplus":
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            model.eval()
            image = convert_from_cv2_to_image(image)
            t = T.Compose([T.Resize((opts.unet_image_size,opts.unet_image_size)), T.ToTensor(), T.Normalize(mean, std)])
            image = t(image)
            image.to(device)
            with torch.no_grad():
                image = image.unsqueeze(0)
                outputs = model(image)
        elif opts.region_detection_model.lower() == "yolo" or (opts.region_ensemble and model_type == "yolo") or opts.table_detection_model.lower() == "yolo" or opts.table_structure_detection_model.lower() == "yolo":
            imgsz = (opts.yolo_image_size,opts.yolo_image_size)
            resized_img = cv2.resize(image, imgsz)
            results = model.predict(resized_img, verbose=False, device=opts.device)
            outputs = results[0]
        elif opts.region_detection_model.lower() == "segformer" or (opts.region_ensemble and model_type == "segformer"):
        
            feature_extractor_inference = SegformerImageProcessor(do_random_crop=False, do_pad=False)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(opts.device)
            model.eval()
            outputs = model(pixel_values=pixel_values)

    elif model_type == "table":
        if opts.table_detection_model.lower() == "maskrcnn":
            outputs = model(image)
        elif opts.table_detection_model.lower() == "yolo":
            imgsz = (opts.yolo_image_size,opts.yolo_image_size)
            resized_img = cv2.resize(image, imgsz)
            results = model.predict(resized_img, max_det = 800, verbose=False, device=opts.device)
            outputs = results[0]

    return outputs, height, width

def get_regions(opts, imagepath, model, OCRclasses, model_type="region"):
    reverse_class = {v: k for k, v in OCRclasses.items()}
    palette = np.array([[OCRclasses[k],OCRclasses[k],OCRclasses[k]] for k in OCRclasses.keys()  ])
    outputs, o_rows, o_cols = predict_image(opts, model, imagepath, model_type)
 
    if opts.region_detection_model.lower() == "maskrcnn":
        scores = [x.item() for x in list(outputs['instances'].scores)]
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs['instances'].pred_boxes]

#       model trained for classes starting from 0 instead of 1, background is reserved for class=0
#       fix it by re-training the model

        labels = [reverse_class[i.item()+1] for i in outputs['instances'].pred_classes]
        mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
        num_instances = mask_array.shape[0]

        mask_array = np.moveaxis(mask_array, 0, -1)

        bmasks = []
        for i in range(num_instances):
            bmasks.append(mask_array[:, :, i:(i + 1)])

        n_boxes, n_labels, n_scores, n_masks = [],[],[],[]
        for i in range(len(labels)):
            m_image = np.float32(bmasks[i])
            m_image = cv2.normalize(m_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            contours, _ = cv2.findContours(m_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            region_candidates = []
            for contour in contours:
                if contour.shape[0] < 4:
                    continue
                if cv2.contourArea(contour) < opts.min_area * o_rows * o_cols:
                    continue
                polygon = geometry.Polygon([point[0] for point in contour])
                reg_coords = np.array(list(polygon.exterior.coords)).astype(np.int_)
                xmax, ymax = reg_coords.max(axis=0)
                xmin, ymin = reg_coords.min(axis=0)

                n_labels.append(labels[i])
                n_boxes.append(boxes[i])
                n_scores.append(scores[i])
                n_masks.append(reg_coords)
        opts.statistics['regions_detected'] += len(n_labels)
        return n_masks, n_boxes, n_labels, n_scores
    elif opts.region_detection_model.lower() == "unet" or opts.region_detection_model.lower() == "unetplusplus":
        r_masks = outputs.cpu().numpy()
        min_mask = np.min(r_masks[0])
        max_mask = np.max(r_masks[0])
        lim_lower = max_mask - (max_mask - min_mask) * opts.unet_score
        lim_higher = max_mask
        n_boxes, n_labels, n_scores, n_masks = [],[],[],[]
        bmasks = []
        for i in range(1,len(opts.region_classes)):
            r_data = r_masks[0][i]
            r_data = cv2.resize(r_data, (o_cols, o_rows))
            #reg_mask = sigmoid(r_data)
            reg_mask = r_data
            reg_mask = (reg_mask > 0.5).astype("uint8") * 255
            bmasks.append(reg_mask)
            contours,_ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            region_candidates = []
            for contour in contours:
                if contour.shape[0] < 4:
                    continue
                if cv2.contourArea(contour) < opts.min_area * o_rows * o_cols:
                    continue
                polygon = geometry.Polygon([point[0] for point in contour])
                reg_coords = np.array(list(polygon.exterior.coords)).astype(np.int_)
                xmax, ymax = reg_coords.max(axis=0)
                xmin, ymin = reg_coords.min(axis=0)

                n_masks.append(reg_coords)
                n_boxes.append([xmin,ymin, xmax, ymax])
                n_labels.append(i)
        nn_scores = [1.0 for l in n_labels]
        nn_labels = [reverse_class[l] for l in n_labels]
        nn_boxes = [[(int(b[0]), int(b[1])), (int(b[2]), int(b[3]))] for b in n_boxes]
    elif opts.region_detection_model.lower() == "yolo" or (opts.region_ensemble and model_type == "yolo"):
        result = outputs.cpu()
        if result.masks != None:
            imgsz = opts.yolo_image_size

            nn_labels = [reverse_class[int(l)] for l in result.boxes.cls.tolist()]
            masks = [m for m in result.masks.xy]
            nn_scores = [x.item() for x in result.boxes.conf]
            n_masks = []
            for mask in masks:
                n_mask = []
                mask_l = mask.tolist()
                for pair in mask_l:
                    n_mask.append([round(pair[0] * o_cols / imgsz), round(pair[1] * o_rows / imgsz)])
                n_masks.append(np.array(n_mask))
            nn_boxes = []

            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                coords = [(round(coords[0] * o_cols / imgsz), 
                           round(coords[1] * o_rows / imgsz)), 
                          (round(coords[2] * o_cols / imgsz),
                           round(coords[3] * o_rows / imgsz))]
                nn_boxes.append(coords)
        else:
            nn_boxes, nn_labels, nn_scores, n_masks = [],[],[],[]
           
    elif opts.region_detection_model.lower() == "segformer" or (opts.region_ensemble and model_type == "segformer"):
        logits = outputs.logits.cpu()
        resized_logits = nn.functional.interpolate(logits,
                size=[o_rows,o_cols], # (height, width)
                mode='bilinear',
                align_corners=False)
        seg = resized_logits.argmax(dim=1)[0]
        labels = np.unique(seg)
        labels = labels[1:]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3\
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        r_map = color_seg[:,:,0]
        masks = r_map == labels[:, None, None]
        n_boxes, n_labels, n_scores, n_masks = [],[],[],[]
        for i,label in enumerate(labels):
            m_image = np.float32(masks[i])
            m_image = cv2.normalize(m_image, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            contours, _ = cv2.findContours(m_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if contour.shape[0] < 4:
                    continue
                if cv2.contourArea(contour) < opts.min_area * o_rows * o_cols:
                    continue
                polygon = geometry.Polygon([point[0] for point in contour])
                reg_coords = np.array(list(polygon.exterior.coords)).astype(np.int_)
                xmax, ymax = reg_coords.max(axis=0)
                xmin, ymin = reg_coords.min(axis=0)
                n_masks.append(reg_coords)
                n_boxes.append([xmin,ymin, xmax, ymax])
                n_labels.append(label)

        nn_scores = [1.0 for l in n_labels]
        nn_labels = [reverse_class[l] for l in n_labels]
        nn_boxes = [[(int(b[0]), int(b[1])), (int(b[2]), int(b[3]))] for b in n_boxes]

#    nn_boxes = list(nn_boxes)
    indices_to_remove = [i for i, item in enumerate(nn_scores) if item < 0.5]
     
    for i in reversed(indices_to_remove):
        n_masks.pop(i)
        nn_boxes.pop(i)
        nn_labels.pop(i)
        nn_scores.pop(i)
    return n_masks, nn_boxes, nn_labels, nn_scores
    
def get_table_predictions(opts, imagepath, model, Tableclasses):
    reverse_class = {v: k for k, v in Tableclasses.items()}
    outputs, o_rows, o_cols = predict_image(opts, model, imagepath, model_type="table")
 
    if opts.table_detection_model.lower() == "maskrcnn":
        scores = [x.item() for x in list(outputs['instances'].scores)]
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs['instances'].pred_boxes]

        labels = [reverse_class[i.item()+1] for i in outputs['instances'].pred_classes]
        mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
        num_instances = mask_array.shape[0]

        mask_array = np.moveaxis(mask_array, 0, -1)

        bmasks = []
        for i in range(num_instances):
            bmasks.append(mask_array[:, :, i:(i + 1)])

        n_boxes, n_labels, n_scores, n_masks = [],[],[],[]
        for i in range(len(labels)):
            m_image = np.float32(bmasks[i])
            m_image = cv2.normalize(m_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            contours, _ = cv2.findContours(m_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            region_candidates = []
            for contour in contours:
                if contour.shape[0] < 4:
                    continue
                if cv2.contourArea(contour) < opts.min_area * o_rows * o_cols:
                    continue
                polygon = geometry.Polygon([point[0] for point in contour])
                reg_coords = np.array(list(polygon.exterior.coords)).astype(np.int_)
                xmax, ymax = reg_coords.max(axis=0)
                xmin, ymin = reg_coords.min(axis=0)

                n_labels.append(labels[i])
                n_boxes.append(boxes[i])
                n_scores.append(scores[i])
                n_masks.append(reg_coords)
        opts.statistics['regions_detected'] += len(n_labels)
        return n_masks, n_boxes, n_labels, n_scores
    elif opts.table_detection_model.lower() == "yolo" or opts.table_structure_detection_model.lower() == "yolo":
        result = outputs.cpu()
        if result.masks != None:
            imgsz = opts.yolo_image_size

            nn_labels = [reverse_class[int(l)] for l in result.boxes.cls.tolist()]
            masks = [m for m in result.masks.xy]
            nn_scores = [x.item() for x in result.boxes.conf]
            n_masks = []
            for mask in masks:
                n_mask = []
                mask_l = mask.tolist()
                for pair in mask_l:
                    n_mask.append([round(pair[0] * o_cols / imgsz), round(pair[1] * o_rows / imgsz)])
                n_masks.append(np.array(n_mask))
            nn_boxes = []

            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                coords = [(round(coords[0] * o_cols / imgsz), 
                           round(coords[1] * o_rows / imgsz)), 
                          (round(coords[2] * o_cols / imgsz),
                           round(coords[3] * o_rows / imgsz))]
                nn_boxes.append(coords)
        else:
            nn_boxes, nn_labels, nn_scores, n_masks = [],[],[],[]

    opts.statistics['tables_detected'] += len(nn_labels)       
    return n_masks, nn_boxes, nn_labels, nn_scores
 
def get_lines(opts, imagepath, model, OCRclasses, model_type="line", area=None):
    reverse_class = {v: k for k, v in OCRclasses.items()}
    outputs, o_rows, o_cols = predict_image(opts, model, imagepath, model_type=model_type, area=area)
    
    if opts.line_detection_model.lower() == "maskrcnn":
        scores = [x.item() for x in list(outputs['instances'].scores)]
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs['instances'].pred_boxes]
       
        labels = [i.item() for i in outputs['instances'].pred_classes]

        labels = [reverse_class[i.item()] for i in outputs['instances'].pred_classes]
        
        mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
        num_instances = mask_array.shape[0]

        mask_array = np.moveaxis(mask_array, 0, -1)

        bmasks = []
        for i in range(num_instances):
            bmasks.append(mask_array[:, :, i:(i + 1)])

        n_boxes, n_labels, n_scores, n_masks = [],[],[],[]
        for i in range(len(labels)):
            m_image = np.float32(bmasks[i])
            m_image = cv2.normalize(m_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            contours, _ = cv2.findContours(m_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            region_candidates = []
            for contour in contours:
                if contour.shape[0] < 4:
                    continue
                if cv2.contourArea(contour) < opts.min_area * o_rows * o_cols:
                    continue
                polygon = geometry.Polygon([point[0] for point in contour])
                reg_coords = np.array(list(polygon.exterior.coords)).astype(np.int_)
                xmax, ymax = reg_coords.max(axis=0)
                xmin, ymin = reg_coords.min(axis=0)

                n_labels.append(labels[i])
                n_boxes.append(boxes[i])
                n_scores.append(scores[i])
                n_masks.append(reg_coords)

        opts.statistics['lines_detected'] += len(n_labels)
        return n_masks, n_boxes, n_labels, n_scores
    elif opts.line_detection_model.lower() == "yolo":
        result = outputs.cpu()
        if result.masks != None:
            imgsz = opts.yolo_image_size

            nn_labels = [reverse_class[int(l)] for l in result.boxes.cls.tolist()]
            masks = [m for m in result.masks.xy]
            nn_scores = [x.item() for x in result.boxes.conf]
            n_masks = []
            for mask in masks:
                n_mask = []
                mask_l = mask.tolist()
                for pair in mask_l:
                    n_mask.append([round(pair[0] * o_cols / imgsz), round(pair[1] * o_rows / imgsz)])
                n_masks.append(np.array(n_mask))
            nn_boxes = []

            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                coords = [(round(coords[0] * o_cols / imgsz), 
                           round(coords[1] * o_rows / imgsz)), 
                          (round(coords[2] * o_cols / imgsz),
                           round(coords[3] * o_rows / imgsz))]
                nn_boxes.append(coords)
        else:
            nn_boxes, nn_labels, nn_scores, n_masks = [],[],[],[]  
    elif opts.line_detection_model.lower() == "segformer":
        palette = np.array([[OCRclasses[k],OCRclasses[k],OCRclasses[k]] for k in OCRclasses.keys()  ])
        logits = outputs.logits.cpu()
        resized_logits = nn.functional.interpolate(logits,
                size=[o_rows,o_cols], # (height, width)
                mode='bilinear',
                align_corners=False)
        seg = resized_logits.argmax(dim=1)[0]
        labels = np.unique(seg)
        labels = labels[1:]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3\
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]
        r_map = color_seg[:,:,0]
        masks = r_map == labels[:, None, None]
        n_boxes, n_labels, n_scores, n_masks = [],[],[],[]
        for i,label in enumerate(labels):
            m_image = np.float32(masks[i])
            m_image = cv2.normalize(m_image, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            contours, _ = cv2.findContours(m_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if contour.shape[0] < 4:
                    continue
                if cv2.contourArea(contour) < opts.min_area * o_rows * o_cols:
                    continue
                polygon = geometry.Polygon([point[0] for point in contour])
                reg_coords = np.array(list(polygon.exterior.coords)).astype(np.int_)
                xmax, ymax = reg_coords.max(axis=0)
                xmin, ymin = reg_coords.min(axis=0)
                n_masks.append(reg_coords)
                n_boxes.append([xmin,ymin, xmax, ymax])
                n_labels.append(label)

        nn_scores = [1.0 for l in n_labels]
        nn_labels = [reverse_class[l] for l in n_labels]
        nn_boxes = [[(int(b[0]), int(b[1])), (int(b[2]), int(b[3]))] for b in n_boxes]
    elif opts.line_detection_model.lower() == "kraken":
        n_labels, n_boxes, n_scores, n_masks = [],[],[],[]
        for dline in outputs['lines']:
            n_labels.append('line')
            n_boxes.append(createBbox(np.array(dline['boundary'])))
            n_scores.append(1.0)
            n_masks.append(np.array(dline['boundary']))
        opts.statistics['lines_detected'] += len(n_labels)
        return n_masks, n_boxes, n_labels, n_scores
    elif opts.region_detection_model.lower() == "yolo" or (opts.region_ensemble and model_type == "yolo"):
        result = outputs.cpu()
        if result.masks != None:
            imgsz = opts.yolo_image_size

            nn_labels = [reverse_class[int(l)] for l in result.boxes.cls.tolist()]
            masks = [m for m in result.masks.xy]
            nn_scores = [x.item() for x in result.boxes.conf]
            n_masks = []
            for mask in masks:
                n_mask = []
                mask_l = mask.tolist()
                for pair in mask_l:
                    n_mask.append([round(pair[0] * o_cols / imgsz), round(pair[1] * o_rows / imgsz)])
                n_masks.append(np.array(n_mask))
            nn_boxes = []

            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                coords = [(round(coords[0] * o_cols / imgsz), 
                           round(coords[1] * o_rows / imgsz)), 
                          (round(coords[2] * o_cols / imgsz),
                           round(coords[3] * o_rows / imgsz))]
                nn_boxes.append(coords)
        else:
            nn_boxes, nn_labels, nn_scores, n_masks = [],[],[],[]
           
    indices_to_remove = [i for i, item in enumerate(nn_scores) if item < 0.5]
     
    for i in reversed(indices_to_remove):
        n_masks.pop(i)
        nn_boxes.pop(i)
        nn_labels.pop(i)
        nn_scores.pop(i)
    opts.statistics['lines_detected'] += len(n_labels)
    return n_masks, nn_boxes, nn_labels, nn_scores
