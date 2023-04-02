import os
from cv2 import imread, cvtColor, imshow, waitKey,COLOR_BGR2GRAY, imwrite

def convert_grayscale(img, img_dir, img_name):
    # convert to grayscale
    img_gs = cvtColor(img, COLOR_BGR2GRAY)
    return img_gs

def save_image(image, image_name, dir_name):
    # save image to specified directory
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        os.chdir(dir_name)

    imwrite(os.path.join(dir_name, image_name), image)

def show_image(img_name, img):
    # display image
    imshow(img_name, img)
    waitKey(1000)

def main():
    cur_dir = os.getcwd()
    img_dir = os.path.join(cur_dir, "2011_09_26\\2011_09_26_drive_0001_sync\\image_02\\data")
    output_dir = os.path.join(img_dir, "output")

    # loop through the images
    for x in os.listdir(img_dir):
        img_path = os.path.join(img_dir, x)
        img = imread(img_path)
        img_gs = convert_grayscale(img, img_dir, x)
        save_image(img_gs, x, output_dir)
        show_image(str(x), img_gs)


if __name__ == "__main__":
    main()

