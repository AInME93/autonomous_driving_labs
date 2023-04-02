import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

dir_path = "C:\\Users\\imran\\Documents\\My Docs\\SLU\\Msc AI\\SEM 3\\CSCI 5930\\Assignments\\2011_09_26_drive_0001_sync"
img_path = os.path.join(dir_path, "image_02\data\\0000000000.png")

img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) # `<opencv_root>/samples/data/blox.jpg`

EXTENSIONS = ['png', 'jpg']

def list_images(dir_path):

    #add all files in the dir to a list
    files = os.listdir(dir_path)

    #extension verification
    files = [f for f in files if f.endswith('.png')]

    return files


# def pos():


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


def orb_feature_extract(img, nfeatures=3000):

    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=30000)

    # compute the descriptors with ORB
    kp, des = orb.detectAndCompute(img, None)
 
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

    return img2, kp, des


def main():

    # specify work directories
    dir_path = "C:\\Users\\imran\\Documents\\My Docs\\SLU\\Msc AI\\SEM 3\\CSCI 5930\\Assignments\\2011_09_26_drive_0001_sync\\image_02\\data"
    output_dir = os.path.join(dir_path, "output")

    # get names of files
    files = list_images(dir_path)

    #read images
    img1 = cv.imread(os.path.join(dir_path, files[0]), cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(os.path.join(dir_path, files[4]), cv.IMREAD_GRAYSCALE)

    # get keypoints and descriptors
    _, kp1, des1 = orb_feature_extract(img1)
    _, kp2, des2 = orb_feature_extract(img2)


    # Create a brute-force matcher and match the descriptors
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Apply distance ratio test to filter out ambiguous matches
    matches = sorted(matches, key = lambda x:x.distance)

    # good_matches = []
    # for m in matches:
    #     if m.distance < 0.7 * matches[0].distance:
    #         good_matches.append(m)

    # Draw the matched keypoints on the images
    match_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img = cv.resize(match_img, (1280,360))

    # Display the result
    cv.imshow('Matches', match_img)
    cv.waitKey(0)
    cv.destroyAllWindows()    

if __name__ == "__main__":
    main()
