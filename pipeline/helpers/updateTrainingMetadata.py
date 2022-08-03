import os
from glob import glob
import pandas as pd

def updateTrainingMetadata(skip_string="skip",proj_dir="../.."):
    """
        Merge .csv annotations in /trainset_tasks into one train_metadata.csv file in /data
        :param str skip_string a string to look for in .csv file names then skip that file if found - used to exclude certain annotation tasks for example
    """

    # get all csv file locations in /trainset_tasks
    csv_file_locs = [file for path, subdir, files in os.walk(proj_dir + "/trainset_tasks") for file in glob(os.path.join(path, "*.csv"))]
    csv_file_locs = [s.replace('\\', '/') for s in csv_file_locs]
    csv_file_locs = [s for s in csv_file_locs if not skip_string in s]

    print("Retrieved training metadata .csv locations: " + str(csv_file_locs))

    # concatenate all csv dataframes
    df = pd.DataFrame()
    df['file'] = "1"

    for csv_loc in csv_file_locs:
        #print(pd.read_csv(csv_loc))
        #if "segment" in csv_loc:
        #df = df.merge(pd.read_csv(csv_loc), on='file', how='outer')
        #else:
        new_df = pd.read_csv(csv_loc)
        df = pd.concat([df, new_df])
        print("Merged " + csv_loc + "...")
    print("Finished merging training metadata...")

    # drop duplicate annotations, may add more sophicated merging options for this later
    df = df.drop_duplicates(subset='file', keep='first')
    print("Dropped duplicates...")

    # ImageAnt annotations always have file field with image name - id is just image name minus .jpg
    df['imageID'] = df['file'].str.replace('.jpg','')
    print("Added imageID column...")

    # fill -1s in NAs or NaNs
    df.fillna(int(-1), inplace=True)

    # write merged training metadata to /data
    df.to_csv(proj_dir + "/data/train_metadata.csv")
    print("Wrote updated training metadata - finished")