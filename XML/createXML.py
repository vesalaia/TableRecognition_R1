"""
PAGE XML creation

some ideas borrowed from https://github.com/lquirosd/P2PaLA 
"""
import logging
import os
import xml.etree.ElementTree as ET
import datetime
import cv2
import numpy as np
import math

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except:
    logging.info(f"Transformers not supported")

from model.inference import get_regions, get_lines, classify_images, classify_cell_type, classify_cell_type_in_batch
from text_recognition.line2text import TRline2Text, recognize_text_in_batches, TRImage2Text
from post_processing.utils import postProcessBox, postProcessLines, on_the_same_line, mergeLines, remove_total_overlaps
from utils.bbox_and_mask import bb_iou, combineMasks, combineMasksIntersection
from utils.viz_utils import plotBBoxes, drawLinePolygon
from post_processing.reading_order import updateRegionRO, updateLineRO, updateCellRO
from utils.page import createBbox
from utils.page import get_lines_area, splitRegion, map_lines_to_region, map_lines_to_cell
from utils.cv_utils import convert_from_cv2_to_image, convert_from_image_to_cv2, get_polygon_from_table, crop_polygons, crop_line_polygons_from_region
                 
from utils.cols_and_rows import detectColumns, detectRows

#blla.segment

from utils.orientation import detect_rotation_angle


# Get current date and time

def ensemble(opts, labels1, boxes1, masks1, scores1, 
                   labels2, boxes2, masks2, scores2):
    tp = []
    fp = []
    fn = []

    shape =(max(1,len(labels1)), max(1,len(labels2)))
    iou_arr = np.zeros(shape)
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            iou_arr[i,j] = bb_iou(b1, b2)
    for i in range(shape[0]):
        ind = np.argmax(iou_arr[i,:])
        if iou_arr[i,ind] > 0.5:
            tp.append([i,ind,2])

    tpinl1 = [l[0] for l in tp]
    tpinl2 = [l[1] for l in tp]
    for ind, l in enumerate(labels1):
        if not ind in tpinl1:
            xind = np.argmax(iou_arr[ind,:])
            if iou_arr[ind,xind] > 0.0:
                tp.append([ind, xind,1])
            else:
                fn.append([ind, l])
    for ind, l in enumerate(labels2):
        if not ind in tpinl2:
            yind = np.argmax(iou_arr[:,ind])
            if iou_arr[yind,ind] == 0.0:   
                fp.append([ind, l])

    n_labels, n_boxes, n_masks, n_scores = [], [], [], []
    for obj in tp:
         if obj[2] == 1:
             n_labels.append(labels1[obj[0]])
             n_boxes.append(boxes1[obj[0]])
             n_masks.append(masks1[obj[0]])
             n_scores.append(scores1[obj[0]])
         elif obj[2] == 2:
             n_labels.append(labels1[obj[0]])
             n_scores.append(scores1[obj[0]])
             if opts.ensemble_combine_method.lower() == "intersection":
                 c_mask = combineMasksIntersection(masks1[obj[0]], masks2[obj[1]])
             elif opts.ensemble_combine_method.lower() == "union":
                 c_mask = combineMasks(masks1[obj[0]], masks2[obj[1]])
             else: # 
                 c_mask = combineMasks(masks1[obj[0]], masks2[obj[1]])

             c_xmax, c_ymax = c_mask.max(axis=0)
             c_xmin, c_ymin = c_mask.min(axis=0)
             n_masks.append(c_mask)
             n_boxes.append([(c_xmax, c_ymax), (c_xmin, c_ymin)])

    for obj in fp:
         n_labels.append(labels2[obj[0]])
         n_boxes.append(boxes2[obj[0]])
         n_masks.append(masks2[obj[0]])
         n_scores.append(scores2[obj[0]])

    return n_labels, n_boxes, n_masks, n_scores
    
def find_corner_points(cell):
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf

    for (x, y) in cell:
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
class XMLData:
    """ Class to process PAGE xml files"""

    def __init__(self, opts, infile, xmlfile, region_classes=None, line_classes=None, line_model="mask r-cnn",
                 tryMerge=False, debug=False,
                 region_renaming=False, reading_order=False, text_recognition=False, combine=True):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.xml = None
        self.imagename = infile
        if not os.path.exists(os.path.dirname(xmlfile)):
            os.makedirs(os.path.dirname(xmlfile))
        self.xmlname = xmlfile
        self.region_classes = opts.region_classes if hasattr(opts, "region_classes") else {}
        self.line_classes = opts.line_classes if hasattr(opts, "line_classes") else {}
        self.tryMerge = tryMerge
        self.combine_lines =  combine
        self.reading_order_update = reading_order
        self.debug = opts.DEBUG
        self.height = 0
        self.width = 0
        self.region_renaming = region_renaming
        self.text_recognition = text_recognition
        self.line_model = line_model

        self.creator = "DL-tools for historians -project/University of Helsinki"
        self.XMLNS = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": " ".join(
                [
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
                ]
            ),
        }

    def new_xml(self):
        """create a new PAGE xml"""

        self.xml = ET.Element("PcGts")
        self.xml.attrib = self.XMLNS
        metadata = ET.SubElement(self.xml, "Metadata")
        ET.SubElement(metadata, "Creator").text = self.creator
        ET.SubElement(metadata, "Created").text = datetime.datetime.today().strftime(
            "%Y-%m-%dT%X"
        )
        ET.SubElement(metadata, "LastChange").text = datetime.datetime.today().strftime(
            "%Y-%m-%dT%X"
        )
        self.page = ET.SubElement(self.xml, "Page")
        self.image = cv2.imread(self.imagename)
        (o_rows, o_cols, _) = self.image.shape
        self.height = o_rows
        self.width = o_cols

        self.page.attrib = {
            "imageFilename": os.path.basename(self.imagename),
            "imageWidth": str(o_cols),
            "imageHeight": str(o_rows),
        }

    def print_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        ET.dump(tree)
        # tree.write(self.filepath, encoding="UTF-8", xml_declaration=True)

    def save_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        tree.write(self.xmlname, encoding="UTF-8", xml_declaration=True)
        logging.info(f"XML file: {self.xmlname} created")

    def _indent(self, elem, level=0):
        """
        Function borrowed from:
            http://effbot.org/zone/element-lib.htm#prettyprint
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _addTextRegions(self, opts, parent, page):
        parent = self.page if parent == None else parent
        t_readingOrder = ET.SubElement(parent, "ReadingOrder")
        t_orderedGroup = ET.SubElement(t_readingOrder, "OrderedGroup")
        t_orderedGroup.attrib = {
            "id": "RO_1",
            "caption": "Regions reading order"
        }
        idx = 0
        for reg in page:
            t_RegRefIdx = ET.SubElement(t_orderedGroup, "RegionRefIndexed")
            t_RegRefIdx.attrib = {
                "index": str(idx),
                "regionRef": reg['id']
            }
            idx += 1
        idx = 0
        for reg in page:
            t_Region = ET.SubElement(self.page, "TextRegion")
            t_Region.attrib = {
                "id": reg['id'],
                "custom": "".join(["readingOrder {index:", str(idx), ";} structure {type:", reg['type'], ";}"])
            }
            t_coords = ET.SubElement(t_Region, "Coords")
            reg_coords = ""
            for pair in reg['polygon']:
                reg_coords = reg_coords + " {},{}".format(pair[0], pair[1])
            t_coords.attrib = {
                "points": reg_coords.strip()
            }
            idx += 1

    def _addTextLines(self, opts, parent, page):
        if self.text_recognition:
            if opts.text_recognize_engine.lower() == "tesseract":
                for idx, reg in enumerate(page['regions']):
                    for line_idx, line in enumerate(reg['lines']):
                        extracted_text = TRline2Text(opts, self.imagename, line['Textline'])
                        opts.statistics['lines_recognized'] += 1
                        
                        reg['lines'][line_idx]['Text'] = extracted_text
            elif opts.text_recognize_engine.lower() == "trocr":
                for idx, reg in enumerate(page['regions']):
                    cropped_images =  crop_line_polygons_from_region(self.imagename, reg)
                    if opts.debug: logging.info(f"Lines: {len(cropped_images)}")
                    recognized_texts = recognize_text_in_batches(opts, cropped_images, opts.trocr_batch_size, opts.text_recognize_model,opts.text_recognize_processor)
                    opts.statistics['lines_recognized'] += len(recognized_texts)
                    for i, text in enumerate(recognized_texts):
                        if opts.debug: logging.info(f"Recognized text: {text}")
                        reg['lines'][i]['Text'] = text


        parent = self.page if parent == None else parent
        t_readingOrder = ET.SubElement(parent, "ReadingOrder")
        t_orderedGroup = ET.SubElement(t_readingOrder, "OrderedGroup")
        t_orderedGroup.attrib = {
            "id": "RO_1",
            "caption": "Regions reading order"
        }
        for idx, reg in enumerate(page['regions']):
            t_RegRefIdx = ET.SubElement(t_orderedGroup, "RegionRefIndexed")
            t_RegRefIdx.attrib = {
                "index": str(idx),
                "regionRef": reg['id']
            }
        for idx, reg in enumerate(page['regions']):
            t_Region = ET.SubElement(self.page, "TextRegion")
            t_Region.attrib = {
                "id": reg['id'],
                "custom": "".join(["readingOrder {index:", str(idx), ";} structure {type:", reg['type'], ";}"])
            }
            t_coords = ET.SubElement(t_Region, "Coords")
            reg_coords = ""
            for pair in reg['polygon']:
                reg_coords = reg_coords + " {},{}".format(pair[0], pair[1])
            t_coords.attrib = {
                "points": reg_coords.strip()
            }
            for line_idx, line in enumerate(reg['lines']):
                t_Line = ET.SubElement(t_Region, "TextLine")
                t_Line.attrib = {
                    "id": line['id'],
                    "custom": "".join(["readingOrder {index:", str(line_idx), ";}"])
                }
                if len(line['Textline']) > 0:
                    text_coords = ""
                    for pair in line['Textline']:
                        text_coords = text_coords + " {},{}".format(pair[0], pair[1])
                    t_Textline = ET.SubElement(t_Line, "Coords")
                    t_Textline.attrib = {
                        "points": text_coords.strip()
                    }

                if len(line['Baseline']) > 0:
                    base_coords = ""   
                    for pair in line['Baseline']:
                        base_coords = base_coords + " {},{}".format(pair[0], pair[1])
                    t_Baseline = ET.SubElement(t_Line, "Baseline").attrib = {"points": base_coords.strip()}

                if len(line['Textline']) > 0 and len(line['Text']) > 0:
                    t_Textequiv = ET.SubElement(t_Line, "TextEquiv")
                    t_Unicode = ET.SubElement(t_Textequiv, "Unicode")
                    t_Unicode.text = line['Text']    
    
    def _processRegions(self, opts, dataset, idx, model, model2=None):
        logging.info(f"Process Regions starting")
        if model != None:
            logging.info(f"Read XML file")
            page = dataset.__getXMLitem__(idx)
            if opts.debug: logging.info(f"{page}")

            logging.info(f"Get region predictions")
            if opts.region_ensemble:
                r_masks_1, r_boxes_1, r_labels_1, r_scores_1 = get_regions(opts, self.imagename, model, 
                                                                           self.region_classes, model_type="yolo")
                r_masks_2, r_boxes_2, r_labels_2, r_scores_2 = get_regions(opts, self.imagename, model2, 
                                                                           self.region_classes, model_type="segformer")

                r_labels, r_boxes, r_masks, r_scores = ensemble( opts, r_labels_1, r_boxes_1, r_masks_1, r_scores_1,
                                                                 r_labels_2, r_boxes_2, r_masks_2, r_scores_2)
            else:
               r_masks, r_boxes, r_labels, r_scores = get_regions(opts, self.imagename, model, self.region_classes)

            if opts.debug:
                logging.info(f"Labels: {len(r_labels)}")
                logging.info(f"Boxes: {len(r_boxes)}")
                logging.info(f"Masks: {len(r_masks)}")

            if self.tryMerge:
                logging.info(f"Try to minimize false positive predictions")
                if opts.merge_type.lower() == "partial":
                    r_masks, r_boxes, r_labels = postProcessBox(opts, self.imagename, r_masks, r_boxes, r_labels,
                                                            opts.merged_region_elements)
                else:
                   r_masks, r_boxes, r_labels = postProcessBox(opts, self.imagename, r_masks, r_boxes, r_labels, mergedElements=None)

            if self.region_renaming:
                logging.info(f"Renaming of elements")
                r_labels = list(map(lambda x: x.replace('marginalia-top', 'marginalia'), r_labels))
            regions = []
            i = 0
            for r_mask_item, r_boxes_item, r_labels_item in zip(r_masks, r_boxes, r_labels):
                reg = {}
                reg['id'] = r_labels_item + "_" + str(i)
                reg['polygon'] = np.array(r_mask_item)
                reg['bbox'] = r_boxes_item
                reg['type'] = r_labels_item
                regions.append(reg)
                i += 1

        if self.reading_order_update:
            logging.info(f"Ensure right reading order for regions")
            newpage = updateRegionRO(opts, regions, opts.RO_region_groups)
        self._addTextRegions(opts, self.page, newpage)
        
    def _addTable(self, opts, parent, table, idx, text_recognition=False):
        parent = self.page if parent == None else parent
        
        t_Table = ET.SubElement(self.page, "TableRegion")
        t_Table.attrib = {
                          "id": table['id'],
                          "custom": "".join(["readingOrder {index:",str(idx),";}"])   
        }
        t_coords = ET.SubElement(t_Table,"Coords")
        table_coords = ""
        for pair in table['polygon']:
            table_coords = table_coords + " {},{}".format(max(0, pair[0]), max(0,pair[1]))
        t_coords.attrib = {
                           "points": table_coords.strip()
        }
        if 'table' in table:      
            for cell in table['table']:
                t_Cell = ET.SubElement(t_Table, "TableCell")
                t_Cell.attrib = {
                                 'id': cell['id'],
                                 'row': str(cell['row']),
                                 'col': str(cell['col']),
                                 'rowSpan': "1",
                                 'colSpan': "1" }                   
                t_coords = ET.SubElement(t_Cell,"Coords")
                cell_corner_points = find_corner_points(cell['polygon'])

                cell_coords = ""
#                for pair in cell['polygon'][::-1]:   # ensure we have the corner points in right order
                for pair in cell_corner_points:
                    cell_coords = cell_coords + " {},{}".format(max(0,pair[0]), max(0,pair[1]))
                t_coords.attrib = {
                                   "points": cell_coords.strip() 
                }
                t_cornerPts = ET.SubElement(t_Cell, "CornerPts")
                t_cornerPts.text = "0 1 2 3"
                if 'lines' in cell:
                    if len(cell['lines']) > 0: 
                        line_idx = 0
                        for line in cell['lines']:
                            t_Line =  ET.SubElement(t_Cell, "TextLine")
                            t_Line.attrib = {
                                             "id": line['id'],
                                             "custom": "".join(["readingOrder {index:",str(line_idx),";}"])
                            }
                            if len(line['Textline']) > 0:
                                text_coords = ""
                                for pair in line['Textline']:
                                    text_coords = text_coords + " {},{}".format(max(0, pair[0]), max(0, pair[1]))
                                t_Textline = ET.SubElement(t_Line,"Coords")
                                t_Textline.attrib = {
                                                     "points": text_coords.strip() 
                                }
 
                            if "Text" in line:
                                t_Textequiv = ET.SubElement(t_Line, "TextEquiv")
                                t_Unicode = ET.SubElement(t_Textequiv, "Unicode")
                                if isinstance(line["Text"], bytes):
                                    t_Unicode.text = str(line["Text"],'utf-8')
                                else:
                                    t_Unicode.text = line["Text"]
                            line_idx += 1                        
        idx += 1
    def _processTables(self, opts, dataset, idx, colmodel, rowmodel):
        logging.info(f"Process Tables starting")
        if colmodel != None and rowmodel != None:
            logging.info(f"Get table predictions")
            c_masks, c_boxes, c_labels, c_scores = get_table_predictions(opts, self.imagename, colmodel)
            r_masks, r_boxes, r_labels, r_scores = get_table_predictions(opts, self.imagename, rowmodel)
            page = get_table_cells(opts, self.imagename, c_masks, c_boxes, c_labels, c_scores, r_masks, r_boxes, r_labels, r_scores)
        self._addTable(opts, self.page, page)

    def _processCellLines(self, opts, dataset, idx, model):
        logging.info(f"Process Table cell lines starting")
        if model != None:
            logging.info(f"Read XML file")
            page = dataset.__getXMLitem__(idx)
            logging.info(f"Get line predictions")
            if opts.line_detection_model == "maskrcnn":
                l_masks, l_boxes, l_labels, l_scores = get_lines(opts, self.imagename, model, self.line_classes, 
                                                                                 model_type="line")
                if opts.debug: logging.info(f"Lines detected: {len(l_boxes)}")
                lines = []
                i = 0
                for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                    if l_labels_item == "line":
                        line = {}
                        line['id'] = "line" + "_" + str(i)
                        line['Textline'] = np.array(l_mask_item)
                        line['bbox'] = createBbox(line['Textline'])
                        lines.append(line)
                        i += 1

                for table in page['regions']:
                    if 'table' in table:
                        if len(table['table']) > 0:
                            for cell in table['table']:
                                new_cell = map_lines_to_cell(opts, cell, lines)

                                if len(new_cell['lines']) > 1:
                                    cell['lines'] = new_cell['lines']
                                else: # if not able to detect multiple lines, then we use the cell coordinates
                                    line = {}
                                    line['id'] = "line" + "_" + str(i)
                                    line['Textline'] = np.array(cell['polygon'])
                                    line['bbox'] = createBbox(line['Textline'])
                                    cell['lines']= [line]
                                    i += 1
                        else:
                             cell['lines'] = []
 
            for r_idx, table in enumerate(page['regions']):
                self._addTable(opts, self.page, table, r_idx)  
        
    def _processCellLineswithCellcoords(self, opts, dataset, idx):
        logging.info(f"Process Table cell with cell coordinates starting")
        logging.info("Read XML file")
        page = dataset.__getXMLitem__(idx)
        for table in page['regions']:
            if 'table' in table:
                if len(table['table']) > 0:
                    for i, cell in enumerate(table['table']):
                        line = {}
                        line['id'] = "line" + "_" + str(i)
                        line['Textline'] = np.array(cell['polygon'])
                        line['bbox'] = createBbox(line['Textline'])
                        cell['lines']= [line]
                else:
                     cell['lines'] = []
        for r_idx, table in enumerate(page['regions']):
            self._addTable(opts, self.page, table, r_idx)  
            
    def _processCellLineswithMultilineOnly(self, opts, dataset, idx, model):
        logging.info(f"Process Table cell lines starting, line detection only for multi-line cells")
        if model != None:
            logging.info(f"Read XML file")
            page = dataset.__getXMLitem__(idx)
            logging.info(f"Get line predictions")
            line_detection_already_done = False
            i = 0
            image_loaded = False
            for table in page['regions']:
                if 'table' in table:
                    if len(table['table']) > 0:
                        if opts.debug: logging.info(f"{self.imagename}")
                        cropped_images =  crop_polygons(self.imagename, table['table'])
                        for ci in cropped_images:
                            if opts.debug: logging.info(f"{ci.size}")
                        cell_types = classify_cell_type_in_batch(opts, cropped_images, opts.cell_classifier)
                        for cell, cell_type in zip(table['table'], cell_types):
                            if cell_type in ['multi-line']:
                                if opts.debug: logging.info(f"Row: {cell['row']} Column: {cell['col']} Cell type:{cell_type}")
                                if not line_detection_already_done:
                                    logging.info(f"Multi-line cell found and running full page line detection")
                                    l_masks, l_boxes, l_labels, l_scores = get_lines(opts, self.imagename, model, self.line_classes, 
                                                                                     model_type="line" )
                                    lines = []

                                    for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                                        if l_labels_item == "line":
                                            line = {}
                                            line['id'] = "line" + "_" + str(i)
                                            line['Textline'] = np.array(l_mask_item)
                                            line['bbox'] = createBbox(line['Textline'])
                                            lines.append(line)
                                            i += 1
                                    line_detection_already_done = True
                                new_cell = map_lines_to_cell(opts, cell, lines)
                                if opts.debug: logging.info(f"{len(new_cell['lines'])} lines detected for cell Row: {cell['row']} Column: {cell['col']}")
                                if len(new_cell['lines']) > 1:
                                    cell['lines'] = new_cell['lines']
                                else: # if not able to detect multiple lines, then we use the cell coordinates
                                    line = {}
                                    line['id'] = "line" + "_" + str(i)
                                    line['Textline'] = np.array(cell['polygon'])
                                    line['bbox'] = createBbox(line['Textline'])
                                    cell['lines']= [line]
                                    i += 1
                            elif cell_type in ['empty','line','same-as']:
                                line = {}
                                line['id'] = "line" + "_" + str(i)
                                line['Textline'] = np.array(cell['polygon'])
                                line['bbox'] = createBbox(line['Textline'])
                                cell['lines']= [line]
                                i += 1
                    else:
                         cell['lines'] = []

        #logging.info(f"{str(page)}")
        for r_idx, table in enumerate(page['regions']):
            self._addTable(opts, self.page, table, r_idx)   
        
    def _processCellTextLines(self, opts, dataset, idx):
        logging.info("Process Table cell text lines starting")
        logging.info("Read XML file")
        page = dataset.__getXMLitem__(idx)
        self._addTable(opts, self.page, page, text_recognition=True)


    def _processLines(self, opts, dataset, idx, model):
        logging.info("Process lines starting")
        page = []
        if model != None:
            logging.info("Read XML file")
            page = dataset.__getXMLitem__(idx)
            if opts.debug: logging.info(f"_processLines: {page}")
            logging.info(f"Get line predictions")
            if self.line_model == "mask r-cnn":
                l_masks, l_boxes, l_labels, l_scores = get_lines(opts, self.imagename, model, self.line_classes)
                if opts.debug:
                    for ll, bb, ss in zip(l_labels, l_boxes, l_scores): logging.info(f"{ll}, {bb}, {ss}")
                    plotBBoxes(self.imagename, l_boxes)
                if self.tryMerge and opts.line_merge:
                    logging.info(f"Try to minimize false positive predictions")
                    l_masks, l_boxes, l_labels = postProcessBox(opts, self.imagename, l_masks, l_boxes, l_labels,
                                                                opts.merged_line_elements)
                lines = []
                i = 0
                for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                    if l_labels_item == "line":
                        line = {}
                        line['id'] = "line" + "_" + str(i)
                        line['Textline'] = np.array(l_mask_item)
                        line['bbox'] = createBbox(line['Textline'])
                        lines.append(line)
                        i += 1
                if opts.debug: logging.info(f"{lines}")
            elif self.line_model == "kraken":
                img = Image.open(self.imagename).convert("RGB")
                baseline_seg = blla.segment(img, model=model)
                lines = []
                i = 0
                for dline in baseline_seg['lines']:
                    line = {}
                    line['id'] = "line" + "_" + str(i)
                    line['Textline'] = np.array(dline['boundary'])
                    line['bbox'] = createBbox(line['Textline'])

                    lines.append(line)
                    i += 1

        if len(page) == 0:
            logging.info(f"Page is empty, assuming empty xml")
            if opts.one_page_per_image:
                logging.info(f"One page per image")
                reg = {}
                reg['id'] = "content_0"
                bbox = get_lines_area(lines, self.width, self.height)
                reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                           (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
                reg['bbox'] = bbox
                reg['type'] = "content"
                page.append(reg)
            else:
                logging.info(f"Two pages per image")
                lines1 = []
                lines2 = []
                page_width = math.trunc(self.width / 2)

                for line in lines:
                    if (line['bbox'][0][0] < page_width) and (line['bbox'][1][0] < page_width):
                        lines1.append(line)
                    else:
                        lines2.append(line)
                reg = {}
                reg['id'] = "content_0"
                bbox = get_lines_area(lines1, page_width, self.height)
                reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                           (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
                reg['bbox'] = bbox
                reg['type'] = "content"
                page.append(reg)
                reg = {}
                reg['id'] = "content_1"
                bbox = get_lines_area(lines2, self.width, self.height)
                reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                           (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
                reg['bbox'] = bbox
                reg['type'] = "content"
                page.append(reg)

        newpage = []
        logging.info(f"Map lines to correct regions")
        for region in page:
            newreg = map_lines_to_region(opts, region, lines)
            newpage.append(newreg)
        logging.info(f"Map lines to correct regions - done")
        if opts.debug: logging.info(f"{newpage}")
        if self.combine_lines:
            angle = detect_rotation_angle(self.imagename)
            if abs(angle) > 0.0: angle = 0.0
            logging.info(f"Combine short lines into longer one")
            for reg in newpage:
                reg['lines'] = postProcessLines(opts, reg['lines'], angle)
            logging.info(f"Combine short lines into longer one - done")
        if self.reading_order_update:
            logging.info(f"Ensure right reading order for regions and lines")
            newpage = updateRegionRO(opts, newpage, opts.RO_region_groups)
            for reg in newpage:
                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextLines(opts, self.page, newpage)

    def _updateLines(self, opts, dataset, idx):
        logging.info(f"Process lines starting")
        page = dataset.__getXMLitem__(idx)
        logging.info(f"Page is empty, assuming empty xml")
        lines = []
        for reg in page:
            for line in reg['lines']:
                lines.append(line)
        logging.info(f"{len(lines)}")
        n_page = []
        if opts.one_page_per_image:
            logging.info(f"One page per image")
            reg = {}
            reg['id'] = "content_0"
            bbox = get_lines_area(lines, self.width, self.height)
            reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                       (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
            reg['bbox'] = bbox
            reg['type'] = "content"
            n_page.append(reg)
        else:
            logging.info(f"Two pages per image")
            lines1 = []
            lines2 = []
            page_width = math.trunc(self.width / 2)

            new_midpoint = page_width
            for line in lines:
                if line['bbox'][0][0] < page_width:
                    X1 = "L"
                else:
                    X1 = "R"
                if line['bbox'][1][0] < page_width:
                    X2 = "L"
                else:
                    X2 = "R"
                if X1 != X2:
                    if new_midpoint < line['bbox'][1][0]:
                        new_midpoint = line['bbox'][1][0]

            for line in lines:
                if (line['bbox'][0][0] <= new_midpoint):
                    lines1.append(line)
                else:
                    lines2.append(line)
            reg = {}
            reg['id'] = "content_0"
            bbox = get_lines_area(lines1, new_midpoint, self.height)
            reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                       (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
            reg['bbox'] = bbox
            reg['type'] = "content"
            n_page.append(reg)
            reg = {}
            reg['id'] = "content_1"
            bbox = get_lines_area(lines2, self.width, self.height)
            reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                       (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
            reg['bbox'] = bbox
            reg['type'] = "content"
            n_page.append(reg)

        newpage = []
        logging.info(f"Map lines to correct regions")
        for region in n_page:
            newreg = map_lines_to_region(opts, region, lines)
            newpage.append(newreg)
        logging.info(f"Map lines to correct regions - done")
        if self.combine_lines:
            angle = detect_rotation_angle(self.imagename)
            if abs(angle) > 0.0: angle = 0.0
            logging.info(f"Combine short lines into longer one")
            for reg in newpage:
                reg['lines'] = postProcessLines(opts, reg['lines'], angle)
            logging.info(f"Combine short lines into longer one - done")
        if self.reading_order_update:
            logging.info(f"Ensure right reading order for regions and lines")
            newpage = updateRegionRO(opts, newpage, opts.RO_region_groups)
            for reg in newpage:
                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextLines(opts, self.page, newpage)

    def _processTextLines(self, opts, dataset, idx):
        logging.info(f"Process Text lines starting")
        page = dataset.__getXMLitem__(idx)

        if self.reading_order_update:
            newpage = {}
            newpage['regions'] = updateRegionRO(opts, page, opts.RO_region_groups)
            
            for reg in newpage['regions']:
                if 'lines' in reg:
                    reg['lines'] = updateLineRO(opts, reg['lines'])
                else:
                    reg['lines'] = []

        self._addTextLines(opts, self.page, newpage)


    def _modifyCoords(self, opts, page):
        
        if self.reading_order_update:
            logging.info(f"Ensure right reading order for regions and lines")
            page = updateRegionRO(opts, page, opts.RO_region_groups)
        
        self._addTextLines(opts, self.page, page)  
        
    def _addTableRegions(self, opts, page, text_recognition=False):
             
        for idx, table in enumerate(page['regions']):
            self._addTable(opts, self.page, table, idx, text_recognition=text_recognition )  
            
    def _textRecognition4Table(self, opts, dataset, idx):
    
        image_path = dataset.__getfullname__(idx)
        page = dataset.__getXMLitem__(idx)
        if opts.text_recognize_engine.lower() == "tesseract":
            if 'regions' in page:
                for reg in page['regions']:
                    if 'table' in reg:
                    #
                        r_rows = reg['rows']
                        r_columns = reg['columns']

                        cropped_images, lines_per_image =  crop_polygons(image_path, reg['table'])
                        results = [""] * len(cropped_images)
                        for idx, c_img in enumerate(cropped_images):
                            results[idx] = TRImage2Text(opts, c_img)
                            opts.statistics['lines_recognized'] += 1

                        for cell in reg['table']:
                            row = cell['row']
                            col = cell['col']
                            cell['Text'] = results[row * r_columns + col]
                                    
        elif opts.text_recognize_engine.lower() == "trocr":
            if 'regions' in page:
                for reg in page['regions']:
                    if 'table' in reg:
                    #
                        r_rows = reg['rows']
                        r_columns = reg['columns']
                        if r_rows <= 1 and r_columns <= 1: 
                            if opts.debug: logging.info(f"Rows: {r_rows} Cols: {r_columns}, empty table")
                            continue

                        cropped_images, line_indices =  crop_polygons(image_path, reg['table'])
                        if opts.debug: logging.info(f"Rows: {r_rows} Cols: {r_columns} Cells: {len(cropped_images)}")
                        #print(f"Lines in images: {line_indices}")
                        nonempty_images, nonempty_indices, nonempty_loc, same_as = classify_images(opts, cropped_images, line_indices, opts.cell_classifier)
                        #
                        # to do 
                        results = [""] * len(cropped_images)
                        for s in same_as:
                            results[s] = '"'
                        #print(f"Nonempty loc: {nonempty_loc}")
                        recognized_texts = recognize_text_in_batches(opts, nonempty_images, opts.trocr_batch_size, opts.text_recognize_model,opts.text_recognize_processor)
                        opts.statistics['lines_recognized'] += len(recognized_texts)

                        for idx, text, loc in zip(nonempty_indices, recognized_texts, nonempty_loc):
                            results[idx] = text
                            
                        if opts.debug: logging.info(f"Table rows: {r_rows} Table columns:{r_columns}")
                        for r,loc in zip(results, line_indices):
                            row = loc[0] // r_columns
                            col = loc[0] % r_columns
                            reg['table'][loc[1]]['lines'][loc[2]]['Text'] = r
                            if opts.debug: logging.info(f"Row: {row} Column:{col} Lines: {loc[2]} Text: {r}")
                        #print(reg)
                        logging.info(f"Updating reading order for table cells")
                        for table in page['regions']:
                            if 'table' in table:
                                if len(table['table']) > 0:
                                   for cell in table['table']:
                                       new_lines =  updateCellRO(opts,cell['lines'])
                                       cell['lines'] = new_lines 

        for idx, table in enumerate(page['regions']):
            self._addTable(opts, self.page, table, idx)  
                        
