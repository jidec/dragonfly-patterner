import os
import pandas as pd
import torch
from torchvision import models, transforms
import numpy as np
import os
import cv2
from sklearn.cluster import AffinityPropagation
from showImages import showImages
from torch import optim, nn
from writeToInferences import writeToInferences

def inferClusterMorphs(image_ids, records_group_col="species",classes=["dorsal"],cluster_image_type="pattern",
                       print_steps=False,show=False,proj_dir="../.."):
    """
        Infer "cluster morphs" for a set of ids, typically grouped by species and view class
        Features are extracted using a pretrained vgg16 ImageNet and clusters in feature space are found
        The intent is for these clusters to correspond to different pattern morphs within a species,
        for example male, female, age, and alive/dead
        Cluster morphs are added to data/inferences.csv

        :param List image_ids: a list of image ids to infer for
        :param str records_group_col: the column in records to use for grouping
        :param List classes: the class groups from inferences (i.e. dorsal, lateral, dorsolateral)
        :param str cluster_image_type: whether to cluster raw segments or patterns
    """

    if (print_steps): print("Starting inferClusterMorphs on " + len(image_ids) + " ids")

    # ids must have patterns or segments, keep only those with patterns or segments (depending on what is specified)
    if cluster_image_type == "segment":
        existing_imgs = os.listdir(proj_dir + "/data/segments")
        existing_img_ids = [s.replace("_segment.jpg", "") for s in existing_imgs]
        image_ids = list(set(existing_img_ids) & set(image_ids))
    elif cluster_image_type == "pattern":
        existing_imgs = os.listdir(proj_dir + "/data/patterns/individuals")
        existing_img_ids = [s.replace("_segment.jpg", "") for s in existing_imgs]
        image_ids = list(set(existing_img_ids) & set(image_ids))
    if(print_steps): print("Removed ids without " + cluster_image_type + "s")

    # load records
    records = pd.read_csv(proj_dir + "/data/inatdragonflyusa_records.csv")
    # get only records rows that are in the provided image ids
    records = records[records["recordID"].isin(image_ids)]
    if (print_steps): print("Got rows from records corresponding to ids")

    # get unique groups (i.e. unique species)
    groups = pd.unique(records.loc[:,records_group_col])

    # load inferences
    infers = pd.read_csv(proj_dir + "/data/inferences.csv")

    # load model
    model = models.vgg16(pretrained=True)
    new_model = FeatureExtractor(model)

    # Change the device to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    new_model = new_model.to(device)

    # Transform the image, so it becomes readable with the model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(512),
        transforms.Resize(448),
        transforms.ToTensor()
    ])
    if (print_steps): print("Prepared feature extractor")

    # allow classes to be none and still loop
    if classes is None:
        classes = [classes]
    if (print_steps): print("Starting loop over groups and classes")
    for g in groups:
        for c in classes:

            # get records of group
            recs = records[records[records_group_col] == g]
            recs_ids = recs['recordID']
            if (print_steps): print("Got records of group " + g)

            if c is not None:
                # get inferences of class
                infers = infers[infers["class"] == c,:]
                infers_ids = infers["imageID"]

            # the ids to use is the intersection
            group_class_ids = set(recs_ids).intersection(set(infers_ids))
            if (print_steps): print("Got records of group " + g)

            # create list of image locations
            img_locs = []
            if cluster_image_type == "segment":
                group_class_names = [n + "_segment.jpg" for n in group_class_ids]
                for n in group_class_names:
                    img_locs.append(cv2.imread(proj_dir + "/data/segments/" + n))
            elif cluster_image_type == "pattern":
                    group_class_names = [n + "_pattern.jpg" for n in group_class_ids]
                    for n in group_class_names:
                        img_locs.append(cv2.imread(proj_dir + "/data/patterns/individuals" + n))
            if (print_steps): print("Retrieved " + cluster_image_type + " locations")

            # will contain the features
            features = []
            imgs = []
            # iterate each image
            for img_loc in img_locs:
                img = cv2.imread(img_loc)
                imgs.append(img)

                # transform the image
                img = transform(img)
                # reshape the image. PyTorch model reads 4-dimensional tensor
                # [batch_size, channels, width, height]
                img = img.reshape(1, 3, 448, 448)
                img = img.to(device)
                # we only extract features, so we don't need gradient
                with torch.no_grad():
                    # extract the feature from the image
                    feature = new_model(img)
                    # convert to NumPy Array, reshape it, and save it to features variable
                    features.append(feature.cpu().detach().numpy().reshape(-1))

            # convert to NumPy Array
            features = np.array(features)

            # cluster
            affprop = AffinityPropagation(affinity="euclidean", damping=0.5).fit(features)
            labels = affprop.labels_

            for cluster_num in list(set(labels)):
                c_imgs = []
                for index, l in enumerate(labels):
                    if l == cluster_num:
                        c_imgs.append(imgs[index])
                showImages(show, c_imgs, titles=None)
            new_infers = pd.DataFrame({'imageID': group_class_ids, 'clusterMorph': labels})
            writeToInferences(new_infers,proj_dir)


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out