import random

import cv2
import numpy as np
from showImages import showImages
import glob
from PIL import Image, ImageOps

# rotation can be:
# a number
# center
# ne_sw

# overlap is a tuple of width and height overlap

# todo
# make spiral
# make rotate to center or corner
# allow full overlap rather than blending
def makeCollage(imgs,n_per_row,resize_wh,white_bg=False,overlap_wh=None,rotation="45",rot_jitter=0,show=True):
    imgs = [cv2.resize(im,resize_wh) for im in imgs]
    print(len(imgs))
    collage = None

    # if int rotation specified, rotate
    if isinstance(rotation, int):
        imgs = [rotate_image(im, rotation + random.randint(-rot_jitter,rot_jitter)) for im in imgs]

    dim_argmax = np.argmax([np.shape(im)[0] * np.shape(im)[1] for im in imgs])
    dim_max = (np.shape(imgs[dim_argmax])[0],np.shape(imgs[dim_argmax])[1])
    #imgs = [cv2.resize(im,dim_max) for im in imgs]
    imgs = [resize_with_padding(im, dim_max) for im in imgs]

    i = 0
    row_imgs = []
    r_index = 0
    while(i < len(imgs)):

        # append image to list
        row_imgs.append(imgs[i])

        # if length of the new row is a multiple of n per row
        if len(row_imgs) % n_per_row == 0:
            print("Adding and resetting row")
            # make a collage of the new row
            if overlap_wh is None:
                row_collage = cv2.hconcat(row_imgs)
            else:
                for index,img in enumerate(row_imgs):
                    if index == 0:
                        row_collage = img
                    else:
                        border_len = resize_wh[0] - overlap_wh[0]
                        #print(border_len)
                        # add right border to row collage
                        row_collage = np.asarray(ImageOps.expand(Image.fromarray(row_collage), border=(0, 0, border_len, 0), fill=(0, 0, 0)))
                        #print(np.shape(row_collage))
                        #cv2.imshow('rc', row_collage)

                        if rotation == "center":
                            #print(r_index)
                            angle = np.arctan2(r_index, index) * 180 / np.pi

                            #print(angle)
                            img = rotate_image(img, angle)
                            img = cv2.resize(img,resize_wh)
                            #cv2.imshow("i",img)
                            #cv2.waitKey(0)


                        # add left border to image to be added to row collage
                        img = np.asarray(ImageOps.expand(Image.fromarray(img), border=(border_len*index, 0, 0, 0), fill=(0, 0, 0)))
                        #cv2.imshow('i', img)
                        #cv2.waitKey(0)
                        #print(np.shape(img))
                        row_collage = cv2.addWeighted(row_collage, 1, img, 1, 0)
                        #row_collage = row_collage[:,0:(np.shape(row_collage)[1])]

                #cv2.imshow("i",row_collage)
                #cv2.waitKey(0)

            # append to the old collage if it exists
            if collage is None:
                collage = row_collage

            else:
                if overlap_wh is None:
                    collage = cv2.vconcat([collage,row_collage])
                else:
                    border_len = resize_wh[1] - overlap_wh[1]
                    v_pad = 10
                    collage = np.asarray(
                        ImageOps.expand(Image.fromarray(collage), border=(0, 0, 0, border_len + v_pad), fill=(0, 0, 0)))

                    #cv2.imshow('c', collage)
                    #cv2.waitKey(0)

                    # add bottom border to row image to be added to collage
                    row_collage = np.asarray(
                        ImageOps.expand(Image.fromarray(row_collage), border=(0, (border_len + v_pad) * r_index, 0, 0), fill=(0, 0, 0)))

                    #cv2.imshow('r', row_collage)
                    #cv2.waitKey(0)

                    collage = cv2.addWeighted(collage, 1, row_collage, 1, 0.1)

                    # extract alpha channel from foreground image as mask and make 3 channels
                    #alpha = row_collage[:, :, 3]
                    #alpha = cv2.merge([alpha, alpha, alpha])

                    # extract bgr channels from foreground image
                    #front = row_collage[:, :, 0:3]
                    #collage = collage[:,:,0:3]
                    #front = row_collage
                    #print(np.shape(row_collage))
                    #print(np.shape(front))
                    #print(np.shape(collage))

                    # blend the two images using the alpha channel as controlling mask
                    #collage = np.where(alpha == (0, 0, 0), collage, front)

                    #collage = cv2.addWeighted(collage, 1, row_collage, 1, 0)
                    #collage[0:row_collage.shape[0], 0:row_collage.shape[1]] = row_collage
                    #Image.Image.paste(Image.fromarray(collage),Image.fromarray(row_collage))
                    #collage = np.asarray(collage)
                    #cv2.imshow("i",collage)
                    #cv2.waitKey(0)

            r_index += 1
            # start a new row
            row_imgs = []
            # add to row counter
        i+=1

    if white_bg:
        # LOL SO BAD
        collage[np.where((collage == [0, 0, 0, 255]).all(axis=2))] = [255, 255, 255, 255]
        collage[np.where((collage == [0, 0, 0, 254]).all(axis=2))] = [255, 255, 255, 255]
        collage[np.where((collage == [0, 0, 0, 253]).all(axis=2))] = [255, 255, 255, 255]
        collage[np.where((collage == [0, 0, 0, 252]).all(axis=2))] = [255, 255, 255, 255]
        collage[np.where((collage == [0, 0, 0, 251]).all(axis=2))] = [255, 255, 255, 255]
    if show:
        cv2.imshow('Collage', collage)
        cv2.waitKey(0)
    return collage

def trans_paste(fg_img,bg_img,alpha=1.0,box=(0,0)):
    fg_img_trans = Image.new("RGBA",fg_img.size)
    fg_img_trans = Image.blend(fg_img_trans,fg_img,alpha)
    bg_img.paste(fg_img_trans,box,fg_img_trans)
    return bg_img

def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    #overlay_image = overlay[..., :overlay.shape[2]]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def resize_with_padding(img, expected_size):
    img = Image.fromarray(img)
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return np.asarray(ImageOps.expand(img, padding))