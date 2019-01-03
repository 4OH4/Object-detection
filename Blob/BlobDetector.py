# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:54:34 2018

Recognise circular-ish objects in images, and extract some metrics

@author: Rupert.Thomas
"""

import cv2 as cv
import numpy as np
import pandas as pd
import os

class BlobDetector:
    
    def __init__(self, **kwargs):
        
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.thresholdStep = 1
        params.minThreshold = 0;
        params.maxThreshold = 255;
         
        # Filter by Area.
        params.filterByArea = False
        params.minArea = 75
        params.maxArea = 250
         
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.9
         
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.75
         
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        
        # Create a detector with the parameters
        ver = (cv.__version__).split('.')
        if int(ver[0]) < 3 :
            self.detector = cv.SimpleBlobDetector(params)
        else : 
            self.detector = cv.SimpleBlobDetector_create(params)
            
    def detect(self, img, show=False):
        # Look for dark contiguous regions
        keypoints = self.detector.detect(img)
        
        if show:
            # Show keypoints
            im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imshow("Keypoints", im_with_keypoints)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
        return keypoints

# Extract blob meta data from the keypoints object
def getMetaData(kp):
    x = [kp[i].pt[0] for i in range(len(kp))]
    y = [kp[i].pt[1] for i in range(len(kp))]
    diameter = [kp[i].size for i in range(len(kp))]
    
    return {'x': x, 'y': y, 'diameter': diameter}
    
#%%
#if __name__ == '__main__':
    
# Process a single test image, show results and get meta data
detector = BlobDetector()

img = cv.imread('./Blob/blob_test_image.png', cv.IMREAD_COLOR)

kp = detector.detect(img, show=True)

blob_df = pd.DataFrame(getMetaData(kp))

#%% Process a folder full of images, and collate results

target_directory = 'data'
filename_mask = 'blob_test_image'
extension = '.png'

detector = BlobDetector()
appended_data = []  # store data in a list and concatenate into table at the end

for filename in os.listdir(target_directory):
    if filename.startswith(filename_mask) and filename.endswith(extension): 
        print('Processing file: %s' % filename)
        img = cv.imread(os.path.join(target_directory, filename), cv.IMREAD_COLOR)
        kp = detector.detect(img, show=False)
        results = pd.DataFrame(getMetaData(kp))
        results['filename'] = filename
        appended_data.append(results)
    else:
        print('Ignoring file: %s' % filename)

results_df = pd.concat(appended_data, axis=0)
results_df.to_csv(os.path.join(target_directory, filename_mask + '_blob_output.csv'))