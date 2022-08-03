import os
import shutil

def removeFromFilenames(direct_dir, remove="_1_med"):
    for index, path in enumerate(os.listdir(direct_dir)):
        joined = os.path.join(direct_dir,path)
        if remove in joined:
            replaced = joined.replace(remove,"")
            os.rename(joined,replaced)
        if index % 1000 == 0:
            print(index)