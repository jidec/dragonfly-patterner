import pandas as pd
import os
import subprocess

# read in all Odonata genera
genera = pd.read_csv("genus_list.csv")
start_index = 41 # genus Stylurus
skip_records = True
skip_images = False

for i in range(start_index,len(genera)):
    genus = genera.iloc[i,1]
    if(not skip_records):
        print("Started getting records for genus " + genus)
        subprocess.Popen('helpers\\get_inat_records.py ' + genus + ' -r --research_only', shell=True).wait()
        #os.system('helpers\\get_inat_records.py ' + genus) #+ ' [-r]')
        print("Finished getting records for genus " + genus)
    if(not skip_images):
        print("Started downloading images for genus " + genus)
        subprocess.Popen('helpers\\download_images.py' + ' -i' + ' helpers/genus_image_records/iNat_images-' + genus + '.csv', shell=True).wait()
                  # + ' [-d \'../../../../all_images\']')
        print("Finished downloading images for genus " + genus)
        # rename images as JPGs
        dirname = "helpers/genus_image_records/iNat_images-" + genus + "-raw_images"
        for i, filename in enumerate(os.listdir(dirname)):
            os.rename(dirname + "/" + filename, dirname + "/" + filename + ".jpg")