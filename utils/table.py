"""
Table detection utilities
"""
import logging
import cv2
import numpy as np
from ultralytics import YOLO
import os
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union, split
from sklearn.cluster import DBSCAN
from model.inference import predict_image

def find_borders_of_table(result):
    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
    for box in result.boxes:
        if box.conf[0].item() >= 0.4 and result.names[box.cls[0].item()] == "cell":
  
            coords = box.xyxy[0].tolist()
            coords = [coords[0] * o_cols/imgsz[0], 
                      coords[1] * o_rows/imgsz[1], 
                      coords[2] * o_cols / imgsz[0],
                      coords[3] * o_rows /imgsz[1]]
            coords = [round(x) for x in coords]
            xmin = min(xmin, coords[0], coords[2])
            xmax = max(xmax, coords[0], coords[2])
            ymin = min(ymin, coords[1], coords[3])
            ymax = max(ymax, coords[1], coords[3])
    return xmin, ymin, xmax, ymax
    
def find_row_and_column_cluster_averages(candidates, maxdist):
    X = np.array(candidates).reshape(-1, 1)
    db = DBSCAN(eps=maxdist, min_samples=3).fit(X)
 
    labels = db.labels_

    clusters = {}
    for label in set(labels):
         cluster_points = X[labels == label].flatten()
         clusters[label] = cluster_points
    cluster_averages = []
    for label, cluster_points in clusters.items():
        if label != -1:
            cluster_averages.append(np.mean(cluster_points))
    return sorted([round(x) for x in cluster_averages])
    
def find_row_and_column_clusters(candidates, maxdist):
    X = np.array(candidates).reshape(-1, 1)
    db = DBSCAN(eps=maxdist, min_samples=3).fit(X)
 
    labels = db.labels_

    clusters = {}
    for label in set(labels):
         cluster_points = X[labels == label].flatten()
         clusters[label] = cluster_points
    return clusters
    
def find_clusters(corners):
    corners= np.array(corners)  # Replace with your 300 points

    col_eps = 20
    row_eps = 20
    min_samples = 5  
    
    x_coords = corners[:, 0].reshape(-1, 1)
    dbscan_x = DBSCAN(eps=col_eps, min_samples=min_samples).fit(x_coords)
    x_labels = dbscan_x.labels_
    x_clusters = [corners[x_labels == label] for label in np.unique(x_labels) if label != -1]

    y_coords = corners[:, 1].reshape(-1, 1)
    dbscan_y = DBSCAN(eps=row_eps, min_samples=min_samples).fit(y_coords)
    y_labels = dbscan_y.labels_
    y_clusters = [corners[y_labels == label] for label in np.unique(y_labels) if label != -1]
    
    return x_clusters, y_clusters
    
def find_line(coordinates):
    x_coords = np.array([coord[0] for coord in coordinates])
    y_coords = np.array([-1 * coord[1] for coord in coordinates])

    slope, intercept = np.polyfit(x_coords, y_coords, 1)

    return slope, intercept

def calculate_iou(box1, box2):
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0  # No overlap

    intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate the area of both bounding boxes
    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def remove_overlapping_regions(regions, overlap_threshold=0.3):
    # Sort regions by area (from largest to smallest)
    def calculate_area(coords):
        x1, y1, x2, y2 = coords
        return (x2 - x1) * (y2 - y1)
    
    regions = sorted(regions, key=lambda x: calculate_area(x['coords']), reverse=True)

    # Keep track of the regions to retain
    reduced_regions = []
    
    # Loop through each region
    for i, region in enumerate(regions):
        keep = True
        for j, kept_region in enumerate(reduced_regions):
            # Check if types are the same and calculate IoU
            if region['type'] == kept_region['type']:
                iou = calculate_iou(region['coords'], kept_region['coords'])
                if iou > overlap_threshold:
                    # If overlap is greater than the threshold, remove the smaller region
                    keep = False
                    logging.info(f"Overlapping table detected, removing")
                    break
        
        if keep:
            reduced_regions.append(region)
    
    return reduced_regions

def detect_tables(opts, model, image_path):
    
    img = cv2.imread(image_path)
    tables = []
    imgsz = (opts.yolo_image_size,opts.yolo_image_size) 

    result, o_rows, o_cols = predict_image(opts, model, image_path, model_type="table")
    for box in result.boxes:
        if box.conf[0].item() >= 0.4:
            el_type = result.names[box.cls[0].item()]
            coords = box.xyxy[0].tolist()
            coords = [coords[0] * o_cols/imgsz[0], 
                      coords[1] * o_rows/imgsz[1], 
                      coords[2] * o_cols / imgsz[0],
                      coords[3] * o_rows /imgsz[1]]
            coords = [round(x) for x in coords]
            tables.append({'type': el_type, 'coords': coords})
#
#   remove tables if these are overlapping
#
    return remove_overlapping_regions(tables)
    
def detect_cells(opts, model, image_path):
    img = cv2.imread(image_path)
    cells = []
    imgsz = (opts.yolo_image_size,opts.yolo_image_size) 
    result, o_rows, o_cols = predict_image(opts, model, image_path, model_type="table")
    for box in result.boxes:
        if box.conf[0].item() >= 0.4 and result.names[box.cls[0].item()] == "cell":

            coords = box.xyxy[0].tolist()
            coords = [coords[0] * o_cols/imgsz[0], 
                      coords[1] * o_rows/imgsz[1], 
                      coords[2] * o_cols / imgsz[0],
                      coords[3] * o_rows /imgsz[1]]
            coords = [round(x) for x in coords]
            cells.append(coords)
            opts.statistics['cells_detected'] += 1

    return cells
    
def calculate_area(x1, y1, x2, y2):
    return abs(x2 - x1) * abs(y2 - y1)

def calculate_intersection_area(table, cell):
    x1 = max(table[0], cell[0])
    y1 = max(table[1], cell[1])
    x2 = min(table[2], cell[2])
    y2 = min(table[3], cell[3])
    
    if x1 < x2 and y1 < y2:
        return calculate_area(x1, y1, x2, y2)
    else:
        return 0

def cells_in_tables(opts, tables, cells, threshold=0.5):

    tables_with_cells = [[] for _ in range(len(tables))]

    for i, table in enumerate(tables):
        table_area = calculate_area(*table['coords'])
        
        for cell in cells:
            intersection_area = calculate_intersection_area(table['coords'], cell)
            cell_area = calculate_area(*cell)
            
            # Check if the majority of the cell's area is within the table
            if intersection_area / cell_area > threshold:
#                print(f"table {i} cell {cell} included {intersection_area / cell_area}")
                tables_with_cells[i].append(cell)
#            else:
#                print(f"table {i} cell {cell} rejected {intersection_area / cell_area}")
    return tables_with_cells

def find_borders_of_table(cells):
    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
    for box_coords in cells:
        xmin = min(xmin, box_coords[0], box_coords[2])
        xmax = max(xmax, box_coords[0], box_coords[2])
        ymin = min(ymin, box_coords[1], box_coords[3])
        ymax = max(ymax, box_coords[1], box_coords[3])
    return xmin, ymin, xmax, ymax
def find_corners_of_table_items(table_items):

    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
    cells_x = []
    cells_y = []

    for item in table_items:
        polygon = item['polygon']
        cells_x.extend(polygon[:, 0])
        cells_y.extend(polygon[:, 1])
    if len(cells_x) != 0 and len(cells_y) != 0: 
        xmin = min(cells_x)
        xmax = max(cells_x)
        ymin = min(cells_y)
        ymax = max(cells_y)
    else:
        xmin, xmax, ymin, ymax = 0,0,0,0
#    for box_coords in cells:
#        xmin = min(xmin, box_coords['polygon'][0], box_coords['polygon'][2])
#        xmax = max(xmax, box_coords['polygon'][0], box_coords['polygon'][2])
#        ymin = min(ymin, box_coords['polygon'][1], box_coords['polygon'][3])
#        ymax = max(ymax, box_coords['polygon'][1], box_coords['polygon'][3])
    return xmin, ymin, xmax, ymax
    
def find_cell_corners_of_table(table, cells):
    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf
    corners = []
    corners.append([table[0], table[1]])
    corners.append([table[2], table[1]])
    corners.append([table[0], table[3]])
    corners.append([table[2], table[3]])
    
    for box_coords in cells:
        corners.append([box_coords[0], box_coords[1]])
        corners.append([box_coords[2], box_coords[1]])
        corners.append([box_coords[0], box_coords[3]])
        corners.append([box_coords[2], box_coords[3]])
    return corners
    
def get_table_cells(table_bbox, rows, columns):
    # Create the table polygon
    table = box(*table_bbox)
#    print(f"{table}")
    
    # Create row and column lines
    row_lines = [LineString(row) for row in rows]
    column_lines = [LineString(col) for col in columns]
#    print(f"{row_lines}")
    # Filter lines that intersect the table
#    intersecting_rows = [row for row in row_lines if row.intersects(table)]
#    intersecting_columns = [col for col in column_lines if col.intersects(table)]

    # Sort row lines by y-coordinate and column lines by x-coordinate
#    intersecting_rows.sort(key=lambda line: line.coords[0][1])
#    intersecting_columns.sort(key=lambda line: line.coords[0][0])
    row_lines.sort(key=lambda line: line.coords[0][1])
    column_lines.sort(key=lambda line: line.coords[0][0])


    
    # Get y-coordinates and x-coordinates from row and column lines
#    y_coords = [line.coords[0][1] for line in intersecting_rows]
#    x_coords = [line.coords[0][0] for line in intersecting_columns]
    y_coords = [line.coords[0][1] for line in row_lines]
    x_coords = [line.coords[0][0] for line in column_lines]
    
    # Create cells
    cells = []
    for i in range(len(y_coords) - 1):
        for j in range(len(x_coords) - 1):
            cell = Polygon([
                (x_coords[j], y_coords[i]),
                (x_coords[j+1], y_coords[i]),
                (x_coords[j+1], y_coords[i+1]),
                (x_coords[j], y_coords[i+1]),
                (x_coords[j], y_coords[i])
            ])

            
            if cell.intersects(table):
                cells.append(list(cell.exterior.coords)[:-1])
    cells.sort(key=lambda cell: (cell[0][1], cell[0][0]))
    return len(y_coords)-1, len(x_coords)-1, cells
    
def calculate_center(corners):
    x_coords = [point[0] for point in corners]
    y_coords = [point[1] for point in corners]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return center_x, center_y

def calculate_topleft_corner(corners):
    x_coords = [point[0] for point in corners]
    y_coords = [point[1] for point in corners]
    topleft_x = min(x_coords)
    topleft_y = min(y_coords)
    return topleft_x, topleft_y
    
def sort_tables_by_topleft_corner(tables, tolerance=100):
    # Calculate center points and add them to the dictionaries
    for table in tables:
        table['topleft'] = calculate_topleft_corner(table['polygon'])

    # Custom sort function with tolerance
    def custom_sort_key(table):
        topleft_x, topleft_y = table['topleft']
        # Apply tolerance to y-coordinate
        adjusted_topleft_x = round(topleft_x / tolerance) * tolerance
        adjusted_topleft_y = round(topleft_y / tolerance) * tolerance
        
        return (adjusted_topleft_y, adjusted_topleft_x)

    # Sort by the custom key
    tables_sorted = sorted(tables, key=custom_sort_key)

    for table in tables_sorted:
        del table['topleft']

    return tables_sorted

def merge_rows (row_heights, y_coords, height):
    threshold = height /30 * 0.4
    merged_rows = []
    i = 0
    while i < len(row_heights):
        if row_heights[i] < threshold:
    # Merge current row with the next one
            merged_y = y_coords[i + 1]
            merged_rows.append(merged_y)
            i += 2  # Skip the next row
        else:
            merged_rows.append(y_coords[i])
            i += 1
    if len(merged_rows) > 0 and len(y_coords) > 0: 
        if merged_rows[-1] < y_coords[-1]:
            merged_rows.append(y_coords[-1])

    return merged_rows


def getTableStructure(opts, image_path, table_predictor, cell_predictor):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    tables = detect_tables(opts, table_predictor, image_path)

    cells = detect_cells(opts, cell_predictor, image_path)

    tables_with_cells = cells_in_tables(opts, tables, cells)
    page = {}
    page['reading_order'] = {}
    page['size'] = (height, width)
    regions = []
    for i, item in enumerate(zip(tables_with_cells, tables)):
        reg = {}
        reg['regiontype']= 'TableRegion'
        reg['id'] = 'Table_' + str(i)
        table_cells = item[0]
        table_type = item[1]['type']
        table = item[1]['coords']
        
        reg['polygon'] = np.array([[table[0],table[1]],
                                   [table[2],table[1]],
                                   [table[2],table[3]],
                                   [table[0],table[3]]])
        reg['rows'] = 0
        reg['columns'] = 0
        reg['table'] = []

        if table_type == "content":
            table_items = []
            x1, y1, x2, y2 = table
            corners = find_cell_corners_of_table(table, table_cells)
            rows = np.array(corners)[:,1]
            columns = np.array(corners)[:,0]
            x_clusters = find_row_and_column_cluster_averages(columns, 20) 
            y_clusters = find_row_and_column_cluster_averages(rows, 20)
            #
            # if the table is skewed then the table structure detection with clustering tends to create too na
            if len(y_clusters) > 0:
#                print(f"Predicted rows: {len(y_clusters)}")
                distances = [y_clusters[i+1] - y_clusters[i] for i in range(len(y_clusters)-1)]
                if opts.debug: 
                    logging.info(f"Clusters: {y_clusters}")
                    logging.info(f"Distances: {distances}")
                est_row_height = int(height/30)
                if opts.debug:
                    logging.info(f"Height: {height} Estimated row height: {est_row_height}")
                    logging.info(f"Predicted small rows: {[x for x in distances if x < (height / 30 * 0.4)]}")
                candidates = merge_rows(distances, y_clusters, height)
                if opts.debug:
                    logging.info(f"Row merge: {len(y_clusters) - len(candidates)} Row candidates: {candidates}")
                y_clusters = candidates
                
            row_lines, column_lines = [], []
            for idx, cluster in enumerate(x_clusters):
                start_point = (cluster,y1)
                end_point = (cluster,y2)
                column_lines.append([start_point, end_point])
            
            for idx, cluster in enumerate(y_clusters):
                start_point = (x1, cluster)
                end_point = (x2, cluster)
                row_lines.append([start_point, end_point])
            max_row, max_col, all_cells = get_table_cells(table, row_lines, column_lines)
            row_id, col_id = 0, 0
            for cell in all_cells:
                cell_item = {}
                t_cell = [[int(point[0]),int(point[1])] for point in cell]
                cell_item['id'] = "c_" + str(row_id) + "_" + str(col_id)
                cell_item['row'] = row_id
                cell_item['col'] = col_id
                cell_item['polygon'] = np.array(t_cell)
                if col_id == max_col - 1:
                    row_id += 1
                    col_id = 0
                else:
                    col_id += 1
                table_items.append(cell_item)
            reg['rows'] = max_row
            reg['columns'] = max_col
            reg['table'] = table_items
            x1, y1, x2, y2 = find_corners_of_table_items(table_items)
            reg['polygon'] = np.array([[x1, y1],
                                       [x2, y1],
                                       [x2, y2],
                                       [x1, y2]])
            opts.statistics['rows_detected'] += max_row + 1
            opts.statistics['columns_detected'] += max_col + 1

        regions.append(reg)
    page['regions'] = sort_tables_by_topleft_corner(regions)
    
    return page
