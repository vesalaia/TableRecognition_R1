"""
Model utils
"""
import cv2
import numpy as np

def get_inference_cfg(opts, config_file_path, checkpoint_url, num_classes, device):
    if opts.package_detectron2:
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog, DatasetCatalog
        from detectron2.utils.visualizer import ColorMode
    else:
        raise ModuleNotFoundError()
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
    cfg.DETECTIONS_PER_IMAGE = 30

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    return cfg

def get_regions(opts, imagepath, model, OCRclasses):
    reverse_class = {v: k for k, v in OCRclasses.items()}

    im = cv2.imread(imagepath)

    outputs = model(im)
    scores = [x.item() for x in list(outputs['instances'].scores)]
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs['instances'].pred_boxes]
    labels = [reverse_class[i.item()] for i in outputs['instances'].pred_classes]

    mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
    num_instances = mask_array.shape[0]

    mask_array = np.moveaxis(mask_array, 0, -1)

    bmasks = []
    for i in range(num_instances):
        bmasks.append(mask_array[:, :, i:(i + 1)])

    masks = []
    for i in range(len(labels)):
        m_image = np.float32(bmasks[i])
        m_image = cv2.normalize(m_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        contours, _ = cv2.findContours(m_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_added = 0
        region_candidates = []
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            if cv2.contourArea(contour) < opts.min_area:
                continue
            convexHull = cv2.convexHull(contour)
            reg_coords = ""
            for pair in convexHull:
                reg_coords = reg_coords + " {},{}".format(pair[0][0], pair[0][1])
            reg_coords = np.array([i.split(",") for i in reg_coords.split()]).astype(np.int_)
            contours_added += 1
            masks.append(reg_coords)

    return masks, boxes, labels, scores
