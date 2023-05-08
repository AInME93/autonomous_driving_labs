import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
from utils import get_calib, set_np_arr
from statistics import mean

# dir_path = "C:\\Users\\imran\\Documents\\My Docs\\SLU\\Msc AI\\SEM 3\\CSCI 5930\\Assignments\\2011_09_26_drive_0001_sync"
# img_path = os.path.join(dir_path, "image_02\data\\0000000000.png")

def list_images(dir_path):

    #add all files in the dir to a list
    files = os.listdir(dir_path)

    #extension verification
    files = [f for f in files if f.endswith('.png')]

    return files

def fast_feature_extract(img, threshold):
    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img,None)

    # sort features based on response value
    kp = sorted(kp, key=lambda x: -x.response)[:200]

    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

    # Print all default params
    fast.setThreshold(threshold)
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    cv.imwrite('fast_true.png', img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
    img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    cv.imwrite('fast_false.png', img3)

def orb_feature_extract(img, nfeatures=500):

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures)

    # compute the descriptors with ORB
    kp, des = orb.detectAndCompute(img, None)
 
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

    return img2, kp, des

def log_keypoints(keypoints, kp_set=set(), exclude_kp=[]):

    for x in keypoints:
        if x not in exclude_kp:
            kp_set.add(x)
            
    return kp_set

def save_keypoint_idx(keypoint_idx, idx, file_name="output.txt"):

    # Open file for writing
    with open("output.txt", "a") as file:
        # Write the elements of the list to a line separated by commas
        line = f"{idx}. {','.join(str(e) for e in keypoint_idx)}"
        file.write(line + "\n")

def filter_distance(matches):

    # Apply distance ratio test to filter out ambiguous matches
    # matches = sorted(matches, key = lambda x:x.distance)

    good_matches = []
    # print(matches)
    for m, n in matches:
        if not m.distance < 0.8 * n.distance:
            good_matches.append(m)

    total_matches = np.size(matches)
    
    # # set the minimum distance as the minimum distance we found between key points
    # min_dist = matches[0].distance
    
    # # set the maximum distance as the maximum distance we found between key points
    # max_dist = matches[total_matches - 1].distance
    
    # # set threshold to find good matches
    # good_matches = []
    # if min_dist < 20:  # You can try different threshold
    #     min_dist = 20

    # for i in range(0, total_matches - 1):
    #     if matches[i].distance >= min_dist and matches[i].distance <= max_dist * 0.66:
    #         good_matches.append(matches[i])

    # # sort good matches
    # good_matches = sorted(good_matches, key=lambda x: x.distance)
    
        # return matches
    return good_matches

def display_matches(img1, kp1, img2, kp2, matches, pause_length=1000):
    # Draw the matched keypoints on the images
    match_img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img = cv.resize(match_img, (1280,360))

    # Display the result
    cv.imshow('Matches', match_img)
    cv.waitKey(pause_length)
    cv.destroyAllWindows()

def feature_match_points(dir_path, files, first_idx, idx_increment):

    #read first image
    img1 = cv.imread(os.path.join(dir_path, files[first_idx]), cv.IMREAD_GRAYSCALE)

    # get keypoints and descriptors
    _, kp1, des1 = orb_feature_extract(img1)

    img2 = cv.imread(os.path.join(dir_path, files[first_idx+idx_increment]), cv.IMREAD_GRAYSCALE)

    _, kp2, des2 = orb_feature_extract(img2)

    # Create a brute-force matcher and match the descriptors
    # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    matches = filter_distance(matches)

    # get indices and values of matched keypoints in both images
    matches_img1 = [kp1[m.queryIdx].pt for m in matches]
    matches_img2 = [kp2[m.trainIdx].pt for m in matches]

    # display_matches(img1, kp1, img2, kp2, matches)

    return matches_img1, matches_img2

def find_R_t(coord1:list, coord2:list, K):

    kp_coord_1 = np.int32(coord1)
    kp_coord_2 = np.int32(coord2)
    print(kp_coord_1, kp_coord_2)

    if kp_coord_1 is None or kp_coord_2 is None:
        raise Exception("MAJOR ERROR")

    F, mask = cv.findFundamentalMat(kp_coord_1, kp_coord_2, cv.FM_8POINT)
    # # Print the fundamental matrix and the number of inliers
    # print("Fundamental matrix:\n", F)
    # print("Number of inliers:", np.sum(mask))
    E, _ = cv.findEssentialMat(kp_coord_1, kp_coord_2, K, cv.RANSAC)

    print(E)
    pts, R, t, mask = cv.recoverPose(E, kp_coord_1, kp_coord_2, K)

    return pts, R, t, mask

# def plotTrajectory():


def main():

    list_matched1, list_matched2 = [] , []
    camera_positions = []

    # specify work directories
    dir_path = "..\\..\\2011_09_26_drive_0001_sync\\image_02\\data"
    calib_path = "..\\..\\2011_09_26_drive_0001_sync\\calib_cam_to_cam.txt"

    calib = get_calib(calib_path)

    # get names of files
    files = list_images(dir_path)
    

    # Get camera intrinsic matrix
    # [ fx_00   0   cx_00 ]
    # [   0   fy_00 cy_00 ]
    # [   0     0     1   ]

    K = set_np_arr((3,3), calib["K_00"].strip().split(" "))

    idx_increment = 2

    idx = 0
    while idx < len(files)-idx_increment:
        try:
            # adjust idx_increment (last argument) to compare with next frame or second frame after and so on
            matches1, matches2 = feature_match_points(dir_path, files, idx, idx_increment) 
            list_matched1.append(matches1)
            list_matched2.append(matches2)
        except IndexError:
            print('End of files')
            break

        idx += idx_increment


    # origin = np.array([0, 0, 0])
    cur_position = np.zeros((3,1))
    cur_rotation = np.eye(3)

    camera_positions.append(cur_position)


    # return Rotation and translation  matrices
    for i in range(len(list_matched1)-1):

        pts, R, t, mask = find_R_t(list_matched1[i], list_matched2[i], K)

        cur_position = R.dot(cur_position) + t

        # print('shape r.current',R.dot(cur_position).shape)
        # print('shape t',t.shape)
        camera_positions.append(cur_position)
        
        # print("Camera positions: ",camera_positions[0])
        # break

    plots = np.array(camera_positions).T
    # print((plots[0][0], plots[0][1]))

    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(plots[0][0], plots[0][1], '-b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_title('Vehicle Trajectory')
    plt.show()



if __name__ == "__main__":
    main()
