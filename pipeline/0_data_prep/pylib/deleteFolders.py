import os
import itertools
import shutil

def deleteFolders(direct_dir):
    for index, path in enumerate(os.listdir(direct_dir)):
        joined = os.path.join(direct_dir,path)
        if(os.path.isdir(joined)):
            shutil.rmtree(joined)
        if(index % 1000 == 0):
            print(index)
