# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import label


def separate(boxes_):
    """separate heat boxes by aspect ratio"""
    def _separate_box(box, axis="x"):
        x1, y1, x2, y2 = box
        
        if axis == "x":
            px = (x1 + x2) / 2
            box1 = np.array([x1, y1, px, y2]).astype(int)
            box2 = np.array([px, y1, x2, y2]).astype(int)
        elif axis == "y":
            py = (y1 + y2) / 2
            box1 = np.array([x1, y1, x2, py]).astype(int)
            box2 = np.array([x1, py, x2, y2]).astype(int)
        return box1, box2
    
    boxes = np.array(boxes_).tolist()
    for box in boxes[:]:
        x1, y1, x2, y2 = box
        w = x2-x1
        h = y2-y1
        
        if w / h >= 1.85:
            print("separation x", w / h)
            box1, box2 = _separate_box(box, axis="x")
            boxes.remove(box)
            boxes.append(box1)
            boxes.append(box2)

        elif h / w >= 1.85:
            print("separation y")
            box1, box2 = _separate_box(box, axis="y")
            boxes.remove(box)
            boxes.append(box1)
            boxes.append(box2)
            
    return boxes


class HeatMap(object):
    def __init__(self, threshold=2):
        self._threshold = threshold
        self._heat_map = None
        self._heat_bin = None
        
    def get_boxes(self, boxes, w, h):
        """
        # Args
            boxes : list of tuple (x1, y1, x2, y2)
                detected boxes
            w : int
            h : int
        
        # Returns
            heat_boxes : list of tuple (x1, y1, x2, y2)
        """

        self._heat_map = np.zeros((h, w)).astype(float)
    
        for box in boxes:
            x1, y1, x2, y2 = box
            self._heat_map[y1:y2, x1:x2] += 1
            
        self._heat_bin = self._get_bin()
        heat_boxes = self._extract_boxes()
        return heat_boxes

    

    def _get_bin(self):
        heat_map_bin = np.zeros_like(self._heat_map)
        heat_map_bin[self._heat_map >= self._threshold] = 255
        return heat_map_bin

    def _extract_boxes(self):
        """
        # Args
            heat_map : ndarray
                binary image
        """
        def _box(ccl_map, car_number):
            # Find pixels with each car_number label value
            nonzero = (ccl_map == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            x1 = np.min(nonzerox)
            y1 = np.min(nonzeroy)
            x2 = np.max(nonzerox)
            y2 = np.max(nonzeroy)
            return (x1, y1, x2, y2)
            
        boxes = []
        ccl_map, n_labels = label(self._heat_bin)
        for car_number in range(1, n_labels+1):
            box = _box(ccl_map, car_number)
            boxes.append(box)
        return boxes


if __name__ == '__main__':
    pass
    
    