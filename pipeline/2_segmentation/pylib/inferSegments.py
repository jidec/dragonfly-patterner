import cv2
import torch
import pandas as pd
import numpy as np
from showImages import showImages
from writeToInferences import writeToInferences
from rotateToVertical import rotateToVertical
from extractHoneSegments import extractHoneSegments

def inferSegments(image_ids, model_name, image_size=344, increase_contrast=False, greyscale=False, part_suffix=None, activation_threshold=0.6, bad_hw_multiplier=4, show=False, print_steps=True, print_details=False,proj_dir="../.."):
    """
        Infer and save segment masks for the specified image_ids using a trained model

        :param List image_ids: the imageIDs (image names) to infer from
        :param str model_name: the name of the model contained in data/ml_models to infer with NOT including the .pt suffix
        :param int image_size: the image dimensions that the model was trained on
        :param str part_suffix:
        :param float activation_threshold: the amount the neuron must be activated to register a pixel as part of the segment
            This should be fine-tuned for each use case
        :param bool show: whether or not to show image processing outputs and intermediates
        :param bool print_steps: whether or not to print processing step info after they are performed
    """

    image_locs = image_ids.copy()

    # turn list of ids into list of locations
    for i in range(0,len(image_locs)):
        image_locs[i] = proj_dir + "/data/all_images/" + image_locs[i] + ".jpg"

    # load the trained model and set it to eval mode
    model = torch.load(proj_dir + "/data/ml_models/" + model_name + ".pt")
    model.eval()
    if (print_steps): print("Loaded model and set to eval mode...")

    # pick cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (print_steps): print("Using device " + str(device))
    if (print_steps): print("Estimated time " + str(1.46 * (len(image_locs)/100)) + " minutes")
    if (print_steps): print("Starting loop through image ids...")
    # for each image location
    for index, img_loc in enumerate(image_locs):
        print(img_loc)
        # read a sample image
        img = cv2.imread(img_loc)

        if(img is None):
            continue

        if greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge((img, img, img))

        if increase_contrast:
            #clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            #img = clahefilter.apply(img)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Applying CLAHE to L-channel
            # feel free to try different values for the limit and grid size:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)

            # merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl, a, b))

            # Converting image from LAB Color model to BGR color spcae
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # save start dims and resize to input
        img_dims = (img.shape[1], img.shape[0])
        start_img = np.copy(img)
        img = cv2.resize(img, (image_size, image_size)) #344

        # transpose to correct shape for model
        img = img.transpose(2, 0, 1).reshape(1, 3, image_size, image_size)  #344 # cracks are 480 x 320
        if (print_details): print("Read, resized, & transposed image...")

        model.to(device)

        # get output from model
        with torch.no_grad():
            a = model((torch.from_numpy(img).type(torch.cuda.FloatTensor) / 255).to(device))
        if (print_details): print("Retrieved output from model...")

        # make mask using activation threshold
        mask = a['out'].cpu().detach().numpy()[0][0] > activation_threshold
        mask = mask.astype(np.uint8)  # convert to an unsigned byte
        mask *= 255
        mask = cv2.resize(mask,img_dims)

        if (print_details): print("Made mask from output...")

        # check if mask is bad
        bad_signifier = 0
        # if mask is all white
        if cv2.countNonZero(mask) == 0:
            bad_signifier = 1
        # if mask is all black
        elif cv2.countNonZero(mask) == mask.shape[0] * mask.shape[1]:
            bad_signifier = 1
        else:
            # if more than 2 or 3 components
            output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
            num_labels = output[0]
            if num_labels > 2:
                bad_signifier = 1

            # if largest component is less than 1% of the image
            stats = output[2]
            cc_areas = stats[:, cv2.CC_STAT_AREA]
            max_area = max(cc_areas)
            if max_area < (0.01 * (mask.shape[0] * mask.shape[1])):
                bad_signifier = 1

            rgb_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            vert = rotateToVertical([rgb_mask])

            if vert is None:
                bad_signifier = 1
            else:
                vert = vert[0]
                h = np.shape(vert)[0]
                w = np.shape(vert)[1]
                if (h / w) < bad_hw_multiplier:
                    bad_signifier = 1


        # get ids (kinda dumb, fix this later to do above)
        id = image_ids[index].split("/")[-1].replace(".jpg","")

        inferences = pd.DataFrame([[id, bad_signifier]], columns=['imageID', 'bad_signifier'])

        writeToInferences(inferences,proj_dir)

        if (print_details): print("Checked if mask is bad and saved in inferences")

        showImages(show,[start_img,mask],["Image","Inferred Mask"])

        dest = ""
        if part_suffix is not None:
            dest = proj_dir + "/data/masks/" + id + "_mask-" + part_suffix + '.jpg'
        else:
            dest = proj_dir + "/data/masks/" + id + "_mask.jpg"

        cv2.imwrite(dest,mask)

        if(print_details): print("Saved to " + dest)

        if index % 100 == 0:
            print("Processed " + str(index))