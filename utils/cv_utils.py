"""
Image Utilities
"""
import logging
import numpy as np
import cv2
from PIL import Image

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)
    
def get_polygon_from_table(table):
    cell_polygons = []
    line_polygons = [] 
    for cell in table:
        poly = cell['polygon']
        poly[poly < 0] = 0
        lines_p = []        
        for line in cell['lines']:
            lines_p.append(line['Textline'])
        cell_polygons.append(poly)
        line_polygons.append(lines_p)
            
    return cell_polygons, line_polygons
    
def get_line_polygons_from_region(region):
    line_polygons = [] 
    if 'lines' in region:
        for line in region['lines']:
            poly = line['Textline']
            poly[poly < 0] = 0       
            line_polygons.append(poly) 
    return line_polygons
    
def crop_polygons(image_path, table):
    # Load the original image
    image = cv2.imread(image_path)
    
    cropped_images = []
    lines_per_image = []
    polygons, line_polygons = get_polygon_from_table(table)
    cidx = 0
    for i, (polygon_points, line_points) in enumerate(zip(polygons, line_polygons)):
        # Create a mask with the same dimensions as the image, initialized to zeros (black)
        if len(line_points) == 1:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_points], 255)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            x, y, w, h = cv2.boundingRect(polygon_points)
            cropped_image = masked_image[y:y+h, x:x+w]
            cropped_images.append(convert_from_cv2_to_image(cropped_image))
            lines_per_image.append([cidx, i, 0])
            cidx += 1
        else:
            for j, lp in enumerate(line_points):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [lp], 255)
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                x, y, w, h = cv2.boundingRect(lp)
                cropped_image = masked_image[y:y+h, x:x+w]
                cropped_images.append(convert_from_cv2_to_image(cropped_image))
                lines_per_image.append([cidx, i, j])
                cidx += 1
    return cropped_images, lines_per_image
    
def crop_line_polygons_from_region(image_path, region ):
    # Load the original image
    image = cv2.imread(image_path)
    
    cropped_images = []
    line_polygons = get_line_polygons_from_region(region)
    for line_points in line_polygons:
        # Create a mask with the same dimensions as the image, initialized to zeros (black)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [line_points], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(line_points)
        cropped_image = masked_image[y:y+h, x:x+w]
        cropped_images.append(convert_from_cv2_to_image(cropped_image))

    return cropped_images
