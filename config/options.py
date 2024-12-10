"""
Configuration
"""

import importlib.util

import configparser
import json
import logging



def package_installed(package_name):
    return importlib.util.find_spec(package_name) != None

class Options():
    def __init__(self, cfg_file):
        required_pckgs = ['detectron2', 'pytesseract', 'kraken', 'ultralytics', 'transformers','segmentation_models_pytorch']
        self.installed_pckgs = []

        for pckg in required_pckgs:
            if package_installed(pckg):
                self.installed_pckgs.append(pckg)

        config = configparser.ConfigParser()
        config.read(cfg_file)

        # TEXT
        # default values for TEXT section

        if "RECOGNIZE" in config:
            logging.info("RECOGNIZE Section configuration")
            self.text_recognize = ""
            if "text_recognize" in config["RECOGNIZE"] : 
                self.text_recognize = config["RECOGNIZE"]["text_recognize"]
            
            self.text_recognize_engine = ""
            if "text_recognize_engine" in config["RECOGNIZE"] :
                self.text_recognize_engine = config["RECOGNIZE"]["text_recognize_engine"]

            self.text_recognize_tesseract_language = "eng"
            if "text_recognize_tesseract_language" in config["RECOGNIZE"] :  
                self.text_recognize_tesseract_language = config["RECOGNIZE"]["text_recognize_tesseract_language"]
            
            self.text_recognize_tesseract_cmd = ""
            if "text_recognize_tesseract_cmd" in config["RECOGNIZE"] :  
                import pytesseract
                self.text_recognize_tesseract_cmd = config["RECOGNIZE"]["text_recognize_tesseract_cmd"]
                pytesseract.pytesseract.tesseract_cmd = config["RECOGNIZE"]["text_recognize_tesseract_cmd"]
            
            self.text_recognize_model_path = ""
            if "text_recognize_model_path" in config["RECOGNIZE"]:
                self.text_recognize_model_path = config["RECOGNIZE"]["text_recognize_model_path"]
            
            self.cell_classifier_path = None
            if self.text_recognize_engine.lower() == "trocr":
                if "cell_classifier_path" in config["RECOGNIZE"]:
                    self.cell_classifier_path = config["RECOGNIZE"]["cell_classifier_path"]
                self.trocr_batch_size = 1
                if "trocr_batch_size" in config["RECOGNIZE"]:
                    self.trocr_batch_size = int(config["RECOGNIZE"]["trocr_batch_size"])
                if "text_recognize_processor_path" in config["RECOGNIZE"]:
                    self.text_recognize_processor_path = config["RECOGNIZE"]["text_recognize_processor_path"]
                else:
                    self.text_recognize_processor_path = "microsoft/trocr-base-handwritten"
                
            self.text_recognize_custom_config = ""
            if self.text_recognize_engine.lower() == "tesseract":
               self.text_recognize_custom_config = r'--oem 3 --psm 7'
 ## 
                
            logging.info(f"RECOGNIZE: text_recognize={self.text_recognize}")
            logging.info(f"RECOGNIZE: text_recognize_engine={self.text_recognize_engine}")
            logging.info(f"RECOGNIZE: text_recognize_tesseract_language={self.text_recognize_tesseract_language}")
            logging.info(f"RECOGNIZE: text_recognize_tesseract_cmd={self.text_recognize_tesseract_cmd}")
            logging.info(f"RECOGNIZE: text_recognize_model_path={self.text_recognize_model_path}")
            logging.info(f"RECOGNIZE: cell_classifier_path={self.cell_classifier_path}")
            logging.info(f"RECOGNIZE: trocr_batch_size={self.trocr_batch_size}")
            logging.info(f"RECOGNIZE: text_recognize_processor_path={self.text_recognize_processor_path}")
            logging.info(f"RECOGNIZE: text_recognize_custom_config={self.text_recognize_custom_config}")
        # Execution
        if "EXECUTION" in config:
            logging.info("Execution Section configuration")

            self.DEBUG = False
            if "DEBUG" in config["EXECUTION"] : 
                self.DEBUG = config["EXECUTION"].getboolean("DEBUG")
                self.debug = config["EXECUTION"].getboolean("DEBUG")
            
            self.device = "cpu"
            if "device" in config["EXECUTION"] : 
                self.device = config["EXECUTION"]["device"]
            logging.info(f"EXECUTION: DEBUG={self.DEBUG}")
            logging.info(f"EXECUTION: device={self.device}")

        if "CLASSES" in config:
            logging.info("Classes Section configuration")

            self.line_classes = {}
            if "line_classes" in config["CLASSES"]: 
                self.line_classes = json.loads(config["CLASSES"]["line_classes"])
        
            self.region_classes = {}
            if "region_classes" in config["CLASSES"]: 
                self.region_classes = json.loads(config["CLASSES"]["region_classes"])
            self.line_classLabels = [x for x in self.line_classes.keys()]
            self.line_num_classes = len(self.line_classLabels) 
            
            self.L_UNKNOWN = len(self.line_classes) - 1
            self.R_UNKNOWN = len(self.region_classes) - 1
        
            self.region_classLabels = [x for x in self.region_classes.keys()]

            self.reverse_line_class = {v:k for k,v in self.line_classes.items()}
            self.reverse_region_class = {v:k for k,v in self.region_classes.items()}
        
            self.merged_line_elements = []
            if "merged_line_elements" in config["CLASSES"]: 
                self.merged_line_elements = json.loads(config["CLASSES"]["merged_line_elements"])

            self.merged_region_elements = []
            if "merged_region_elements" in config["CLASSES"]: 
                self.merged_region_elements = json.loads(config["CLASSES"]["merged_region_elements"])

            self.merge_type = "full"
            if "merge_type" in config["CLASSES"]: 
                self.merge_type = config["CLASSES"]["merge_type"]

            self.RO_line_groups = []
            if "RO_line_groups" in config["CLASSES"]: 
                self.RO_line_groups = json.loads(config["CLASSES"]["RO_line_groups"])

            self.RO_region_groups = []
            if "RO_region_groups" in config["CLASSES"]: 
                self.RO_region_groups = json.loads(config["CLASSES"]["RO_region_groups"])

            self.one_page_per_image = True
            if "one_page_per_image" in config["CLASSES"]: 
                self.one_page_per_image = config["CLASSES"].getboolean("one_page_per_image")

            logging.info(f"CLASSES: line_classes={self.line_classes}")
            logging.info(f"CLASSES: region_classes={self.region_classes}")
            logging.info(f"CLASSES: merged_line_elements={self.merged_line_elements}")
            logging.info(f"CLASSES: merged_region_elements={self.merged_region_elements}")
            logging.info(f"CLASSES: merge_type={self.merge_type}")
            logging.info(f"CLASSES: RO_line_groups={self.RO_line_groups}")
            logging.info(f"CLASSES: RO_region_groups={self.RO_region_groups}")
            logging.info(f"CLASSES: one_page_per_image={self.one_page_per_image}")

# line parameters
        if "LINE" in config:
            logging.info("Line Section configuration")
            self.overlap_threshold = 0.1 
            if "overlap_threshold" in config["LINE"]: 
                self.overlap_threshold = float(config["LINE"]["overlap_threshold"])

            self.line_level_multiplier = 0.2
            if "line_level_multiplier" in config["LINE"]: 
                self.line_level_multiplier = float(config["LINE"]["line_level_multiplier"])

            self.line_merge_limit = 10 
            if "line_merge_limit" in config["LINE"]: 
                self.line_merge_limit = float(config["LINE"]["line_merge_limit"])
        
            self.line_merge = False 
            if "line_merge" in config["LINE"]: 
                self.line_merge = config["LINE"].getboolean("line_merge")
        
            self.dist_limit = 0.1
            if "dist_limit" in config["LINE"]: 
                self.dist_limit = float(config["LINE"]["dist_limit"])
        
            self.default_x_offset = 10
            if "default_x_offset" in config["LINE"]: 
                self.default_x_offset = float(config["LINE"]["default_x_offset"])
        
            self.default_y_offset = 10
            if "default_y_offset" in config["LINE"]: 
                self.default_y_offset = float(config["LINE"]["default_y_offset"])
        
            self.default_y_offset_multiplier = 0.9 
            if "default_y_offset_multiplier" in config["LINE"]: 
                self.default_y_offset_multiplier = float(config["LINE"]["default_y_offset_multiplier"])
        
            self.baseline_sample_size = 200
            if "baseline_sample_size" in config["LINE"]: 
                self.baseline_sample_size = float(config["LINE"]["baseline_sample_size"])
        
            self.baseline_offset_multiplier = 0.1 
            if "baseline_offset_multiplier" in config["LINE"]: 
                self.baseline_offset_multiplier = float(config["LINE"]["baseline_offset_multiplier"])
        
            self.line_boundary_margin = 10
            if "line_boundary_margin" in config["LINE"]: 
                self.line_boundary_margin = int(config["LINE"]["line_boundary_margin"])
        
            self.min_line_heigth = 20
            if "min_line_heigth" in config["LINE"]: 
                self.min_line_heigth = int(config["LINE"]["min_line_heigth"])

            self.min_line_width = 20 
            if "min_line_width" in config["LINE"]: 
                self.min_line_width = int(config["LINE"]["min_line_width"])
            
            self.line_detection_model = None
            if "line_detection_model" in config["LINE"]: 
                self.line_detection_model = config["LINE"]["line_detection_model"]
                
            self.cell_line_detection = "cell"
            self.cell_classifier_path = None
            if "cell_line_detection" in config["LINE"]: 
                self.cell_line_detection = config["LINE"]["cell_line_detection"]

            if self.cell_line_detection == "multiline":
                self.cell_classifier_path = None
                if "cell_classifier_path" in config["LINE"]:
                    self.cell_classifier_path = config["LINE"]["cell_classifier_path"]

            self.line_config_path = None
            if "line_config_path" in config["LINE"]: 
                self.line_config_path = config["LINE"]["line_config_path"]

            self.line_model_path = None
            if "line_model_path" in config["LINE"]: 
                self.line_model_path = config["LINE"]["line_model_path"]
        
            self.kraken_model_path = None
            if "kraken_model_path" in config["LINE"]: 
                self.kraken_model_path = config["LINE"]["kraken_model_path"]
                
            self.min_area = 0.0005
            if "min_area" in config["LINE"]: 
                self.min_area = float(config["LINE"]["min_area"])

            self.classifier_path = ""
            if "cell_classifier_path" in config["LINE"]:
                self.cell_classifier_path = config["LINE"]["cell_classifier_path"]
            self.small_objects = True
            
            logging.info(f"LINE: one_page_per_image={self.one_page_per_image}")
            logging.info(f"LINE: overlap_threshold={self.overlap_threshold}")
            logging.info(f"LINE: line_level_multiplier={self.line_level_multiplier}")
            logging.info(f"LINE: line_merge_limit={self.line_merge_limit}")
            logging.info(f"LINE: line_merge={self.line_merge}")
            logging.info(f"LINE: dist_limit={self.dist_limit}")
            logging.info(f"LINE: default_x_offset={self.default_x_offset}")
            logging.info(f"LINE: default_y_offset={self.default_y_offset}")
            logging.info(f"LINE: default_y_offset_multiplier={self.default_y_offset_multiplier}")
            logging.info(f"LINE: baseline_sample_size={self.baseline_sample_size}")
            logging.info(f"LINE: baseline_offset_multiplier={self.baseline_offset_multiplier}")
            logging.info(f"LINE: line_boundary_margin={self.line_boundary_margin}")
            logging.info(f"LINE: min_line_heigth={self.min_line_heigth}")
            logging.info(f"LINE: min_line_width={self.min_line_width}")
            logging.info(f"LINE: line_detection_model={self.line_detection_model}")
            logging.info(f"LINE: cell_line_detection={self.cell_line_detection}")
            if self.cell_line_detection in ['line', 'multilineonly']:
                logging.info(f"LINE: cell_classifier_path={self.cell_classifier_path}")
            logging.info(f"LINE: line_config_path={self.line_config_path}")
            logging.info(f"LINE: line_model_path={self.line_model_path}")
            logging.info(f"LINE: kraken_model_path={self.kraken_model_path}")
            logging.info(f"LINE: min_area={self.min_area}")
            logging.info(f"LINE: small_objects={self.small_objects}")
    
#region detection
        if "REGION" in config:
            logging.info("Region Section configuration")

            self.region_ensemble = False
            if "region_ensemble" in config["REGION"]: 
                self.region_ensemble = config["REGION"].getboolean("region_ensemble")

            self.ensemble_model_path1 = ""
            if "ensemble_model_path1" in config["REGION"]: 
                self.ensemble_model_path1 = config["REGION"]["ensemble_model_path1"]

            self.ensemble_model_path2 = ""
            if "ensemble_model_path2" in config["REGION"]: 
                self.ensemble_model_path2 = config["REGION"]["ensemble_model_path2"]

            self.ensemble_combine_method = "union"
            if "ensemble_combine_method" in config["REGION"]: 
                self.ensemble_combine_method = config["REGION"]["ensemble_combine_method"]

            self.region_detection_model = ""
            if "region_detection_model" in config["REGION"]: 
                self.region_detection_model = config["REGION"]["region_detection_model"]

            self.region_model_path = ""
            if "region_model_path" in config["REGION"]: 
                self.region_model_path = config["REGION"]["region_model_path"]

            self.region_config_path = ""
            if "region_config_path" in config["REGION"]: 
                self.region_config_path = config["REGION DETECTION"]["region_config_path"]

            self.region_num_classes = len(self.region_classLabels)

            self.unet_image_size = 1024
            if "unet_image_size" in config["REGION"]: 
                self.unet_image_size = int(config["REGION"]["unet_image_size"])

            self.unet_score = 0.2
            if "unet_score" in config["REGION"]: 
                self.unet_score = float(config["REGION"]["unet_score"])

            self.min_area = 0.0005
            if "min_area" in config["REGION"]: 
                self.min_area = float(config["REGION"]["min_area"])

            self.yolo_image_size = 640
            if "yolo_image_size" in config["REGION"]: 
                self.yolo_image_size = int(config["REGION"]["yolo_image_size"])

            self.segformer_image_size = 1024
            if "segformer_image_size" in config["REGION"]: 
                self.segformer_image_size = int(config["REGION"]["segformer_image_size"])
            self.small_objects = False
            
            logging.info(f"REGION: region_ensemble={self.region_ensemble}")
            logging.info(f"REGION: ensemble_model_path1={self.ensemble_model_path1}")
            logging.info(f"REGION: ensemble_model_path2={self.ensemble_model_path2}")
            logging.info(f"REGION: ensemble_combine_method={self.ensemble_combine_method}")
            logging.info(f"REGION: region_detection_model={self.region_detection_model}")
            logging.info(f"REGION: region_model_path={self.region_model_path}")
            logging.info(f"REGION: region_config_path={self.region_config_path}")
            logging.info(f"REGION: region_num_classes={self.region_num_classes}")
            logging.info(f"REGION: unet_image_size={self.unet_image_size}")
            logging.info(f"REGION: unet_score={self.unet_score}")
            logging.info(f"REGION: min_area={self.min_area}")
            logging.info(f"REGION: yolo_image_size={self.yolo_image_size}")
            logging.info(f"REGION: segformer_image_size={self.segformer_image_size}")
            logging.info(f"REGION: small_objects={self.small_objects}")

# table detection
        if "TABLE" in config:
            logging.info("Table Section configuration")
            self.table_detection_model = ""
            if "table_detection_model" in config["TABLE"]: 
                self.table_detection_model = config["TABLE"]["table_detection_model"]

            self.table_detection_model_path = ""
            if "table_detection_model_path" in config["TABLE"]: 
                self.table_detection_model_path = config["TABLE"]["table_detection_model_path"]

            self.table_structure_detection_model = ""
            if "table_structure_detection_model_path" in config["TABLE"]: 
                self.table_structure_detection_model_path = config["TABLE"]["table_structure_detection_model_path"]

            self.table_column_detection_model_path = ""
            if "table_column_model_path" in config["TABLE"]: 
                self.table_column_detection_model_path = config["TABLE"]["table_column_model_path"]

            self.table_row_detection_model_path = ""
            if "table_row_model_path" in config["TABLE"]: 
                self.table_row_detection_model_path = config["TABLE"]["table_row_model_path"]

            self.table_classes = {}
            if "table_classes" in config["TABLE"]: 
                self.table_classes = json.loads(config["TABLE"]["table_classes"])

            self.table_classLabels = [x for x in self.table_classes.keys()]
 
            self.reverse_table_class = {v:k for k,v in self.table_classes.items()}
 
            self.table_num_classes = len(self.table_classLabels)

            self.table_line_detection_model = ""
            if "table_line_detection_model" in config["TABLE"]: 
                self.table_line_detection_model = config["TABLE"]["table_line_detection_model"]

            self.table_line_detection_model_path = ""
            if "table_line_detection_model_path" in config["TABLE"]: 
                self.table_line_detection_model_path = config["TABLE"]["table_line_detection_model_path"]
        
            self.yolo_image_size = 640
            if "yolo_image_size" in config["TABLE"]: 
                self.yolo_image_size = int(config["TABLE"]["yolo_image_size"])
                
            logging.info(f"TABLE: table_detection_model={self.table_detection_model}")
            logging.info(f"TABLE: table_detection_model_path={self.table_detection_model_path}")
            logging.info(f"TABLE: table_structure_detection_model_path={self.table_structure_detection_model_path}")
            logging.info(f"TABLE: table_column_detection_model_path={self.table_column_detection_model_path}")
            logging.info(f"TABLE: table_row_detection_model_path={self.table_row_detection_model_path}")
            logging.info(f"TABLE: table_classes={self.table_classes}")
            logging.info(f"TABLE: table_line_detection_model={self.table_line_detection_model}")
            logging.info(f"TABLE: table_line_detection_model_path={self.table_line_detection_model_path}")
            logging.info(f"TABLE: yolo_image_size={self.yolo_image_size}")

        self.statistics = {'files_processed': 0,
                           'regions_detected': 0,
                           'tables_detected': 0,
                           'columns_detected': 0,
                           'rows_detected': 0,
                           'cells_detected': 0,
                           'lines_detected': 0,
                           'lines_recognized': 0,
                           'cell_count': 0}
 
