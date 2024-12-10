"""
Text recognition utilities
"""
import logging
try:
    import pytesseract as pyt
except:
    logging.info(f"Tesseract not supported")
    
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import cv2
from model.inference import classify_images, process_batches
from utils.cv_utils import convert_from_cv2_to_image, convert_from_image_to_cv2, get_polygon_from_table, crop_polygons

def text_recognition(opts, model, processor, batch):
    pixel_values = processor(batch, return_tensors='pt').pixel_values.to(opts.device)
    with torch.no_grad():
         generated_ids = model.generate(pixel_values)
         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)    
    return generated_text
    
def recognize_text_in_batches(opts, images, batch_size, model, processor):

    results = []

    for batch in process_batches(images, opts.trocr_batch_size):
        batch_results = text_recognition(opts, model, processor, batch)
        results.extend(batch_results)

    return results


def TRImage2Text(opts, img, model=None, processor=None):
    if opts.text_recognize_engine == "tesseract":
        try:
            text = pyt.image_to_string(img, config=opts.text_recognize_custom_config,lang=opts.text_recognize_tesseract_language)
            return text
        except:
            return " "
    elif opts.text_recognize_engine == "trocr":
        pixel_values = processor(img, return_tensors='pt').pixel_values.to(opts.device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
         
    else:
         raise NotImplementedError()

def TRline2Text(opts, imgfile, coords, model=None, processor=None):

    img = Image.open(imgfile).convert("RGB")
    w, h = img.size
    x1, y1 = coords.min(axis=0)
    x2, y2 = coords.max(axis=0)
    if (x2>x1) and (y2>y1):
        if opts.DEBUG: logging.info(f'##  box=[{x1},{y1},{x2},{y2}]')
        cropped_image = img.crop((x1,y1, x2,y2))
        if opts.text_recognize_engine.lower() == "tesseract":
            text = TRImage2Text(opts, cropped_image).split("\n")
            ltext = " "
            for l in text:
                if len(l) > len(ltext):
                    ltext = l
            ltext.replace("\n"," ")
            if opts.DEBUG: logging.info(f"Lines: {len(text)} {text} {len(ltext)} {ltext}")
            return ltext
        elif opts.text_recognize_engine.lower() == "trocr":
            text = TRImage2Text(opts, cropped_image, model, processor)
            if opts.DEBUG: logging.info(f"Lines: {len(text)} {text}")
            return text
            
