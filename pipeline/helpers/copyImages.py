import os
import shutil
import pandas as pd

# skeleton - doesn't work yet
def copyImages(source,dest,ids=None):
    """
        Copy images from source to dest, with option to specify image ids
    """

    # get all .jpgs from source as a pd series
    files = pd.Series([f for f in os.listdir(source) if '.jpg' in f]) #if os.isfile(os.join(source, f))
    file_image_ids = pd.Series([f.replace('.jpg','') for f in files])
    # filter for ids
    files = files[file_image_ids.isin(ids)]
    # create src and dest file locs
    source_files = source + "/" + files
    print(source_files)
    dest_files = dest + "/" + files
    print(dest_files)

    for source, dest in tuple(zip(source_files, dest_files)):
        shutil.copy(source,dest)