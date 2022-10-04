import os
from glob import glob
import pandas as pd

def updateTrainingMetadata(skip_string="skip",proj_dir="../.."):
    """
        Merge .csv annotations in /trainset_tasks into one train_metadata.csv file in /data
        Train metadata columns have 0 for false, 1 for true, or -1 for undefined
        :param str skip_string a string to look for in .csv file names then skip that file if found - used to exclude certain annotation tasks for example
    """

    # get all csv file locations in /trainset_tasks
    csv_file_locs = [file for path, subdir, files in os.walk(proj_dir + "/trainset_tasks") for file in glob(os.path.join(path, "*.csv"))]
    csv_file_locs = [s.replace('\\', '/') for s in csv_file_locs]
    csv_file_locs = [s for s in csv_file_locs if not skip_string in s]

    print("Retrieved training metadata .csv locations: " + str(csv_file_locs))
    class_df = pd.DataFrame()
    seg_df = pd.DataFrame()

    # concatenate all csv dataframes
    for index, csv_loc in enumerate(csv_file_locs):
        print("Merging " + csv_loc + "...")

        new_df = pd.read_csv(csv_loc)
        new_df['file'] = new_df['file'].str.replace('.jpg', '')
        new_df['imageID'] = new_df['file'].str.replace('_mask', '')
        new_df = new_df.drop(columns=['file'])

        if "seg" in csv_loc:
            #df_ID_list = df['imageID'].unique().tolist()
            #df = df.append(new_df.loc[~(new_df['imageID'].isin(df_ID_list))], sort=False)
            #df = df.append(new_df,sort=False)
            seg_df = pd.concat([seg_df,new_df])
            #new_df = new_df.drop_duplicates(subset='imageID', keep='first')
            #df = pd.merge(df, new_df, how='inner', on=['imageID'])
            #df = df.drop_duplicates(subset='imageID', keep='first')

            #if "seg" in csv_loc:
            #    df = new_df.merge(df, how='left', left_on=['imageID'], right_on=['imageID'])
            #else:
            #df = pd.concat([df,new_df])
        else:
            class_df = pd.concat([class_df, new_df])
        print("Merged " + csv_loc + "...")
    print("Finished merging training metadata...")

    # drop duplicate annotations, may add more sophicated merging options for this later
    seg_df = seg_df.drop_duplicates(subset='imageID', keep='first')
    class_df = class_df.drop_duplicates(subset='imageID', keep='first')
    #print("Dropped duplicates...")

    # ImageAnt annotations always have file field with image name - id is just image name minus .jpg
    #df['imageID'] = df['file'].str.replace('.jpg','')
    #print("Added imageID column...")
    df = pd.merge(seg_df,class_df,how="outer",on="imageID")
    # fill -1s in NAs or NaNs
    df.fillna(int(-1), inplace=True)

    # write merged training metadata to /data
    #seg_df.to_csv(proj_dir + "/data/seg_metadata.csv", index=False)
    #class_df.to_csv(proj_dir + "/data/class_metadata.csv", index=False)
    df.to_csv(proj_dir + "/data/train_metadata.csv",index=False)
    print("Wrote updated training metadata - finished")