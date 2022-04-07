import pandas as pd
import os
import subprocess
import time
from datetime import datetime

def downloadGenusImages(start_index,end_index,skip_records=False,skip_images=False,pylib_root=''):
    # read in all Odonata genera
    genera = pd.read_csv("pylib/genus_list.csv")
    #start_index = 41 # genus Stylurus

    pylib_root_cmd = pylib_root + '\\'

    # for every genus from start_index to end_index
    for i in range(start_index,end_index):
        # get the genus name
        genus = genera.iloc[i,1]
        if(not skip_records):
            # get and save records for a genus
            print("Started getting records for genus " + genus)
            subprocess.Popen(pylib_root_cmd + 'helpers\\get_inat_records.py ' + genus + ' -r --research_only', shell=True).wait()
            #os.system('helpers\\get_inat_records.py ' + genus) #+ ' [-r]')
            print("Finished getting records for genus " + genus)

            # split genus records into chunks such that a max of ~4.5 gb of images are downloaded per hour
            dirname = pylib_root + '/helpers/genus_image_records/iNat_images-' + genus + '-records_split'
            if(not os.path.isdir(dirname)):
                os.mkdir(dirname)
            j = 1
            for i, chunk in enumerate(pd.read_csv(pylib_root + '/helpers/genus_image_records/iNat_images-' + genus + '.csv', chunksize=7500)):
                #chunk.to_csv('../tmp/split_csv_pandas/chunk{}.csv'.format(i), index=False)
                chunk.to_csv(dirname + '/' + str(j) + '.csv', index=False)
                j += 1
            print("Finished splitting genus records into chunks")

        if(not skip_images):
            print("Started downloading images for genus " + genus)
            # download each record chunk, waiting an hour between
            for record_chunk in os.listdir(pylib_root + '/helpers/genus_image_records/iNat_images-' + genus + '-records_split'):
                subprocess.Popen(pylib_root_cmd + 'helpers\\download_images.py' + ' -i' + ' ' + pylib_root + '/helpers/genus_image_records/iNat_images-' + genus + '.csv', shell=True).wait()
                          # + ' [-d \'../../../../all_images\']')
                print("Finished downloading images for genus " + genus)
                # rename images as JPGs
                dirname = pylib_root + "/helpers/genus_image_records/iNat_images-" + genus + "-raw_images"
                os.mkdir(dirname)
                for i, filename in enumerate(os.listdir(dirname)):
                    os.rename(dirname + "/" + filename, dirname + "/" + filename + ".jpg")

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Waiting one hour starting at " + current_time)
                time.sleep(3600)
