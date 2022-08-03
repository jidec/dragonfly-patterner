import os
import shutil

def deleteIfContains(direct_dir, contains, if_not_contains=False):
    for index, path in enumerate(os.listdir(direct_dir)):
        joined = direct_dir + "/" + path
        print(index)
        if if_not_contains:
            if(contains not in joined):
                os.remove(joined)
        else:
            if (contains in joined):
                os.remove(joined)