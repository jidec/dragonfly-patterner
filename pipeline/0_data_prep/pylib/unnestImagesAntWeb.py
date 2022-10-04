import os
import shutil
import os.path

def unnestImagesAntWeb(direct_dir):
    all_files = []
    #for root, _dirs, files in itertools.islice(os.walk(direct_dir), 1, None):
    #    for filename in files:
    #        all_files.append(os.path.join(root, filename))
    #print("Finished getting filenames, starting move")
    #for index, filename in enumerate(all_files):
    #    shutil.move(filename, direct_dir)
     #   if index % 1000 == 0:
     #       print(index)
    # get all dirs in direct dir
    dirs = [f for f in os.listdir(direct_dir) if not os.path.isfile(os.path.join(direct_dir, f))]
    print("Got dirs")
    print("Looping")
    # for each dir
    for index, d in enumerate(dirs):
        if index % 1000 == 0:
            print(index)
        # join
        path = os.path.join(direct_dir,d)
        # get files
        files = os.listdir(path)
        # move each file (only moving files in dirs)
        for f in files:
            if not os.path.exists(os.path.join(direct_dir,f)):
                shutil.move(os.path.join(path,f), direct_dir)
