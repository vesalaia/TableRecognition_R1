"""
Postprocessing
"""
import cv2
import numpy as np
import copy
import math
import shapely
from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon
from shapely.errors import TopologicalError
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos


def make_valid(ob):

    if ob.is_valid:
        return ob
    try:
        ret = geom_factory(lgeos.GEOSMakeValid(ob._geom))
    except:
        ret = ob
    return ret


def postProcessBox(opts, imagepath, masks, boxes, labels, mergedElements):
    img = cv2.imread(imagepath)
    img = np.array(img)
    n_img = np.array(img)

    (o_rows, o_cols, _) = n_img.shape
    #    print(labels)
    #    labels = target['labels'].tolist()
    #    boxes = target['boxes'].tolist()
    #    masks = target['masks'].to("cpu").numpy()
    n_masks = []
    for i in range(len(labels)):
        n_masks.append(masks[i])

    dist_limit = 0
    n_boxes = copy.deepcopy(boxes)
    n_labels = copy.deepcopy(labels)

    for regs in mergedElements:
        need2Continue = True

        while need2Continue:
            need2Continue, n_boxes, n_labels, n_masks = mergeRegions(opts, n_boxes, n_labels, n_masks, regs[0], regs[1],
                                                                     regs[2])

    return n_masks, n_boxes, n_labels


def postProcessLines(opts, lines, angle):
    n_lines = copy.deepcopy(lines)
    need2Continue = True
    while need2Continue:
        need2Continue, n_lines = mergeLines(opts, n_lines, angle)

    return n_lines


def on_the_same_line(opts, line_1, line_2, angle):
    x_max_1, y_max_1 = line_1['Textline'].max(axis=0)
    x_min_1, y_min_1 = line_1['Textline'].min(axis=0)
    x_max_2, y_max_2 = line_2['Textline'].max(axis=0)
    x_min_2, y_min_2 = line_2['Textline'].min(axis=0)
    rotation_offset = math.trunc(math.sin(math.radians(angle)) * abs(x_min_2 - x_max_1))
    midpoint_1 = y_min_1 + (y_max_1 - y_min_1) / 2
    midpoint_2 = y_min_2 + (y_max_2 - y_min_2) / 2 + rotation_offset
    line_level_diff = int((y_max_1 + y_max_2 - y_min_1 - y_min_2) * opts.line_level_multiplier)

    if abs(midpoint_2 - midpoint_1) < line_level_diff:
        return True
    else:
        return False


def mergeLines(opts, lines, angle):
    #
    # To Do:
    #
    #    print(lines)
    for i in range(len(lines)):
        for j in range(len(lines)):
            if j <= i:
                continue
            #            print("Line: {} Line length: {} {}".format(i, len(lines[i]),lines[i]))
            #            print("Line: {} Line length: {} {}".format(j, len(lines[j]),lines[j]))
            if on_the_same_line(opts, lines[i], lines[j], angle):
                x_max_1, y_max_1 = lines[i]['Textline'].max(axis=0)
                x_min_1, y_min_1 = lines[i]['Textline'].min(axis=0)
                x_max_2, y_max_2 = lines[j]['Textline'].max(axis=0)
                x_min_2, y_min_2 = lines[j]['Textline'].min(axis=0)
                midpoint_1 = y_min_1 + (y_max_1 - y_min_1) / 2
                midpoint_2 = y_min_2 + (y_max_2 - y_min_2) / 2
                x_offset1 = opts.default_x_offset
                x_offset2 = opts.default_x_offset
                y_offset1 = math.trunc((y_max_1 - y_min_1 + 1) / 2 * opts.default_y_offset_multiplier)
                y_offset2 = math.trunc((y_max_2 - y_min_2 + 1) / 2 * opts.default_y_offset_multiplier)

                if opts.DEBUG: print("Y-diff: {} X-diff: {}".format(y_max_2 - y_max_1, x_max_2 - x_max_1))
                poly_1 = make_valid(Polygon(lines[i]['Textline']))
                poly_2 = make_valid(Polygon(lines[j]['Textline']))
                try:
                    if (x_max_1 < x_max_2):
                        poly_x = Polygon([(x_max_1 - x_offset1, midpoint_1 - y_offset1),
                                          (x_min_2 + x_offset2, midpoint_2 - y_offset2),
                                          (x_min_2 + x_offset2, midpoint_2 + y_offset2),
                                          (x_max_1 - x_offset1, midpoint_1 + y_offset1)])
                        #                        poly_x = Polygon([(x_max_1,y_min_1),(x_min_2, y_min_2),(x_min_2,y_max_2),(x_max_1,y_max_1)])
                        poly_x = poly_x.union(poly_1)
                        poly_x = poly_x.union(poly_2)
                    else:
                        #                        poly_x = Polygon([(x_max_2,y_min_2),(x_min_1, y_min_1),(x_min_1,y_max_1),(x_max_2,y_max_2)])
                        poly_x = Polygon([(x_max_2 - x_offset1, midpoint_2 - y_offset2),
                                          (x_min_1 + x_offset1, midpoint_1 - y_offset1),
                                          (x_min_1 + x_offset1, midpoint_1 + y_offset1),
                                          (x_max_2 - x_offset2, midpoint_2 + y_offset2)])
                        poly_x = poly_x.union(poly_1)
                        poly_x = poly_x.union(poly_2)
                    if type(poly_x) == shapely.geometry.polygon.Polygon:
                        lines[i]['Textline'] = np.array(list(poly_x.exterior.coords)).astype(int)
                        lines[i]['Baseline'] = createBaseline_new(opts, lines[i]['Textline'])
                        lines[i]['bbox'] = createBbox(lines[i]['Textline'])
                        lines.pop(j)
                        return True, lines
                    else:
                        continue
                except TopologicalError:
                    print("TopologicalError detected: Lines: {} and {}".format(lines[i]['id'], lines[j]['id']))
                    # print("Line: {} {} ".format(i, lines[i]))
                    # print("Line: {} {} ".format(j, lines[j]))
                    continue

    return False, lines

