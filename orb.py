import cv2
import os
import time
import concurrent.futures
from multiprocessing import Pool, cpu_count
import psutil
# import cProfile
# import re
# cProfile.run('re.compile("foo|bar")')
sample = cv2.imread('SOCOFing/Altered/Altered-Easy/1__M_Left_little_finger_CR.BMP')
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(sample, None)
def compute_keypoints_descriptors(file):
    fingerprint_image = cv2.imread(os.path.join('SOCOFing/Real/', file))
    kp2, des2 = orb.detectAndCompute(fingerprint_image, None)
    kp2_info = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp2]
    return file, kp2_info, des2



def knn_match(args):
    fil, kp2_info, des2 = args
    
    kp2 = [cv2.KeyPoint(x, y, _size, _angle, _response, _octave, _class_id) 
           for (x, y), _size, _angle, _response, _octave, _class_id in kp2_info]
    
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,    # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks=1)   # or pass empty dictionary
    matches = cv2.FlannBasedMatcher(index_params,search_params).knnMatch(des1, des2, k=2)
    match_point = []
    for p_q in matches:
        if len(p_q) == 2:
            p, q = p_q
            if p.distance < 0.3 * q.distance:
                match_point.append(p)

    keypoints = min(len(kp1), len(kp2))
    score = len(match_point) / keypoints * 100

    return fil, score

# def main():
if __name__ == '__main__':
    process_time = time.time()

    #NOTE kp1 is list of keypoints and des is numpy array of shape (number_of_keypoints, 128)
    # ggdf = cv2.drawKeypoints(sample, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('keypoints.jpg', ggdf) // to draw the keypoints of the fingerprint image


    all_kp2_des2 = []

    with Pool(cpu_count()) as p:
        all_kp2_des2=  p.map(compute_keypoints_descriptors, os.listdir('SOCOFing/Real/'))


    end_process_time = time.time()
    # if all_kp2_des2 size is 6000, how can i make it 18000 by duplicating the data


    gg = all_kp2_des2
    all_kp2_des2.extend(all_kp2_des2)
    # # all_kp2_des2.extend(all_kp2_des2)
    # all_kp2_des2.extend(gg)
    best_score = 0
    best_match = None
    knn_start = time.time()

    # with concurrent.futures.ProcessPoolExecutor() as p:
    # count number of process
    
    with Pool() as p:
        print('Number of CPU:', cpu_count())
        results = p.map(knn_match, all_kp2_des2)

    knn_end = time.time()
    print("Number of fingers we ran search on: ",len(results))
    loop_time_start = time.time()
    for fil, score in results:
        if score > best_score:
            best_score = score
            best_match = fil
    loop_time_end = time.time() 
        

    print('Best Score:', best_score)
    print('Filename:', best_match)
    print('Knn Time (Actual time that machine will take):', knn_end - knn_start)
    print('Image Process Time (We will pre-compute it):', end_process_time - process_time)
    print('Loop Time:', loop_time_end - loop_time_start)
    print('Total Time:', time.time() - process_time)
