import os
import pandas as pd

def fixSameObsImageNames(genus):
    """
        Renames images by adding numbers to file names in genus record for multiples of one observation (i.e. 1031230_1,1031230_2...)

        :param str genus: the genus name to fix image names for
    """
    target_dir = '../../data/misc/genus_download_records/iNat_images-' + genus + '-records_split/'

    # for every split record csv
    for filename in os.listdir(target_dir):
        df = pd.read_csv(target_dir + filename)
        names = list(df['obs_id'])
        print(names)
        new_names = []
        # make list of new names, renaming images of the same observation
        for i, v in enumerate(names):
            totalcount = names.count(v)
            count = names[:i].count(v)
            print("Count" + str(count))
            print(totalcount)
            new_names.append(v + str(count + 1) if totalcount > 1 else v)
        df['file_name'] = new_names
        print(names)
        print(new_names)
        df.to_csv(target_dir + filename,mode='w+')
