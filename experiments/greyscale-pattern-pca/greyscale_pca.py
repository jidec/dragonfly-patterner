from glob import iglob
import pandas as pd
import matplotlib as plt
import cv2

faces = pd.DataFrame([])
for path in iglob():
    img = cv2.imread(path,cv2.IMREAD_GREYSCALE)
    face = pd.Series(img.flatten(), name=path)
    faces = faces.append(face)

fig, axes = plt.subplots(10, 10, figsize=(9, 9),
                        gridspec_kw = dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.iloc[i].values.reshape(112, 92), cmap="gray")

from sklearn.decomposition import PCA
#n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
faces_pca = PCA(n_components=0.8)
faces_pca.fit(faces)
fig, axes = plt.subplots(2,10,figsize=(9,3),
 gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
 ax.imshow(faces_pca.components_[i].reshape(112,92),cmap="gray")