from keras import utils
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from sklearn.cluster import AffinityPropagation
import pandas as pd

utils.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variables
imdir = "../example_clade_2"
targetdir = "../out"
number_clusters = 3

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = utils.load_img(imagepath, target_size=(224, 224))
    img_data = utils.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

print(featurelist)

# Clustering
affprop = AffinityPropagation(affinity="euclidean", damping=0.5).fit(np.array(featurelist))
# kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

# Copy images renamed by cluster
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(affprop.labels_):
    print("    Copy: %s / %s" %(i, len(affprop.labels_)), end="\r")
    shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")