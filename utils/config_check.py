import logging

def configuration_ok(opts, task):
    all_ok = False
    if task.lower() == "init" or task.lower() == "i":
        all_ok = True
    elif task.lower() in ["text", "json"]:
        all_ok = True
    elif task.lower() in ["detectregion", "dr", "region"]:
        if hasattr(opts, 'region_detection_model'):
            if opts.region_detection_model == "maskrcnn":
                if opts.region_model_path != None and opts.region_config_path != None and "detectron2" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"Mask R-CNN required for region detection, but not properly configured")
            elif opts.region_detection_model == "yolo":
                if opts.region_model_path != None and "ultralytics" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"YOLO required for region detection, but not properly configured")
            elif opts.region_detection_model == "unet":
                if opts.region_model_path != None and "segmentation_models_pytorch" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"Unet required for region detection, but not properly configured")
            elif opts.region_detection_model == "segformer":
                if opts.region_model_path != None and "transformers" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"Segformer required for region detection, but not properly configured")
            else: 
                all_ok = False
                logging.error(f"Error in region detection configuration")
        else:
            all_ok = False
            logging.error(f"Task is '{task}', but region detection model is not defined")
    elif task.lower() in ["detectline", "dl", "line"]:
        if hasattr(opts, 'line_detection_model'):
            if opts.line_detection_model == "maskrcnn":
                if opts.line_model_path != None and opts.line_config_path != None  and "detectron2" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"Mask R-CNN required for line detection, but not properly configured")
            elif opts.line_detection_model == "yolo":
                if opts.line_model_path != None and "ultralytics" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"YOLO required for line detection, but not properly configured")
            elif opts.line_detection_model == "kraken":
                if opts.line_model_path != None and "kraken" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"Kraken required for line detection, but not properly configured")
            else:
                all_ok = False
                logging.error(f"Error in line detection configuration")
        else:
            all_ok = False
            logging.error(f"Task is '{task}' required for line detection, but not properly configured")

    elif task.lower() in ["recognizetext", "rt", "recognize"]:
        if hasattr(opts, 'text_recognize_engine'):
            if opts.text_recognize_engine == "tesseract":
                if opts.text_recognize_tesseract_cmd != None and "pytesseract" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"Tesseract required for text recognition, but not properly configured")
 
            elif opts.text_recognize_engine == "trocr":
                if opts.text_recognize_model_path != None and "transformers" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"TrOCR required for text recognition, but not properly configured")
            else:
                all_ok = False
                logging.error(f"Error in text recognition configuration")
        else:
            all_ok = False
            logging.error(f"Task is '{task}' required for text recognition, but not properly configured")

    elif task.lower() in ["cellrecognize", "cr"]:
        if hasattr(opts, 'text_recognize_engine'):
            if opts.text_recognize_engine == "tesseract":
                if opts.text_recognize_tesseract_cmd != None and "transformers" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"Tesseract required for table text recognition, but not properly configured")
            elif opts.text_recognize_engine == "trocr":
                if opts.text_recognize_model_path != None and "transformers" in opts.installed_pckgs:          
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"TrOCR required for table text recognition, but not properly configured")
            else:
                all_ok = False
                logging.error(f"Error in table text recognition configuration")
        else:
            all_ok = False
            logging.error(f"Task is '{task}' required for table text recognition, but not properly configured")

    elif task.lower() in ["update", "u"]:
        all_ok = True
    elif task.lower() in ["table", "t"]:
        if hasattr(opts, 'table_detection_model'):
            if opts.table_detection_model == "yolo":
                if opts.table_detection_model_path != None and opts.table_structure_detection_model_path != None and "ultralytics" in opts.installed_pckgs:
                    all_ok = True
                else:
                    all_ok = False
                    logging.error(f"YOLO required for table detection, but not properly configured")
            else:
                all_ok = False
                logging.error(f"Error in table detection configuration")
        else:
            all_ok = False
            logging.error(f"Task is '{task}' required for table cell detection, but not properly configured")
    elif task.lower() in ["cell", "c"]:
        all_ok = True
    elif task.lower() in ["csv", "json", "text"]:
        all_ok = True

    return all_ok
