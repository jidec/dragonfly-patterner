import cv2
import os

def blendAligned(imgs,show=False):
    # set dir containing discretized, aligned, size normalized images
    # dir = "data/segments/discretized_aligned"
    dir = "images"
    img_files = os.listdir(dir)
    img_files = []
    img_files.append("4155610_198_vert.png")
    img_files.append("56568289_564_vert.png")
    img_resize_size = 400

    imgs = []
    for file_name in img_files:
        img = cv2.imread(os.path.join(dir,file_name))
        if img is not None:
            img = cv2.resize(img,(img_resize_size,img_resize_size))
            imgs.append(img)


    # loop through image list and blend all
    for i, img in enumerate(imgs):
        if i == 0:
            first_img = img
            continue
        else:
            second_img = img
            second_weight = 1/(i+1)
            first_weight = 1 - second_weight
            first_img = cv2.addWeighted(first_img, first_weight, second_img, second_weight, 0)

    out = first_img
    cv2.imshow("blended", out)
    cv2.waitKey(0)