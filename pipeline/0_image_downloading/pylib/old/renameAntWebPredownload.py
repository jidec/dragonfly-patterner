import os
import shutil

def fixAntWebPredownload(target_dir):


    # Define the source and destination path
    source = "Desktop/content/waste/"
    destination = "Desktop/content/"

    subfolders = [f.path for f in os.scandir(target_dir + "/") if f.is_dir()]

    for folder in subfolders:
        # get files in subfolder
        files = os.listdir(folder)
        for file in files:
            file_name = os.path.join(source, file)
            shutil.move(file_name, target_dir)

    # delete files not containing proper tags
    filenames = os.listdir(target_dir)
    filenames = [f for f in filenames if not f.contains("_med")]
    for f in filenames:
        os.remove(target_dir + "/" + f)

    # get rid of images without tags
    filenames = os.listdir(target_dir)

    # add AW- prefix
    new_filenames = ["AW-" + f for f in filenames]

    # remove .._med
    new_filenames = [f.replace(".._med","") for f in new_filenames]

    # replace underscore with dash
    new_filenames = [f.replace("_", "-") for f in new_filenames]

    # rename all
    for index, f in filenames:
        f = target_dir + "/" + f
        new_f = target_dir + "/" + new_filenames[index]
        os.rename(f,new_f)