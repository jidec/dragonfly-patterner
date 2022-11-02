import pandas as pd
import os
import time
from datetime import datetime
import shutil
from getiNatRecords import getiNatRecords
from downloadImages import downloadImages
from pathlib import Path

def downloadiNatGenusImages(start_index,end_index,split_start=1, redownload=False, skip_records=False,skip_images=False,full_size=True,proj_dir='../..'):
    """
       Download iNat images by genus using the genus_list.csv created using a different method,
       This method downloads and splits records then referring to those records to download in parallel
       Images are always saved in proj_root/data/all_images folder
       Based on Brian Stucky's iNat downloader, his code modified to fit this use case is held in the /helpers folder

       :param int start_index: the genus index from data/genus_list to start at
       :param int end_index: the genus index from data/genus_list to end at
       :param bool split_start: specify which split record chunk to start at (starting at 1)
       :param bool skip_records: skip creating records if records are already created
       :param bool skip_images: skip downloading images - used for testing
       :param str proj_root: the path to the project folder
    """

    # read in all genera
    genera = pd.read_csv(proj_dir + "/data/other/genus_list.csv")
    #start_index = 41 # genus Stylurus

    pylib_root_cmd = proj_dir.replace("/","\\") + '\\pipeline\\0_image_downloading\\pylib\\'

    # for every genus from start_index to end_index
    for i in range(start_index,end_index):
        # get the genus name
        genus = genera.iloc[i,1]
        if(not skip_records):
            # get and save records for a genus
            print("Started getting records for genus " + genus)
            #shell_cmd = pylib_root_cmd + 'helpers\\get_inat_records.py ' + genus + ' -r --research_only'
            #print(shell_cmd)
            # add full size param if specified
            #if full_size == True:
            #    shell_cmd = shell_cmd + ' -f --full_size'
            #subprocess.Popen(shell_cmd, shell=True).wait()
            #os.system('helpers\\get_inat_records.py ' + genus) #+ ' [-r]')
            getiNatRecords(genus=genus,research_only=True,full_size=full_size,proj_dir=proj_dir)
            print("Finished getting records for genus " + genus)

            # if not redownloading, remove already downloaded images from records
            if not redownload:
                print("Filtering out already downloaded images")
                existing_imgnames = os.listdir(proj_dir + '/data/all_images')
                existing_imgnames = [i.replace('INAT-', '') for i in existing_imgnames]
                records = pd.read_csv(proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '.csv')
                print("Total images: " + str(records.shape[0]))
                mask = records["img_url"].isin(existing_imgnames)
                records = records[~mask]
                print("Undownloaded images: " + str(records.shape[0]))
                records.to_csv(proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '.csv',index=False,mode='w+')

            # split genus records into chunks such that a max of ~4.5 gb of images are downloaded per hour
            # dirname = proj_root + '/pipeline/0_image_downloading/pylib/helpers/genus_download_records/iNat_images-' + genus + '-records_split'
            dirname = proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '-records_split'
            if(not os.path.isdir(dirname)):
                os.mkdir(dirname)
            j = 1
            for i, chunk in enumerate(pd.read_csv(proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '.csv', chunksize=7500)): #proj_root + '/data/other/genus_download_records/iNat_images-'
                #chunk.to_csv('../tmp/split_csv_pandas/chunk{}.csv'.format(i), index=False)
                chunk.to_csv(dirname + '/' + str(j) + '.csv', index=False)
                j += 1
            print("Finished splitting genus records into chunks")

        if(not skip_images):
            print("Started downloading images for genus " + genus)
            # download each record chunk, waiting an hour between
            for c, record_chunk in enumerate(os.listdir(proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '-records_split')):
                if c >= split_start - 1:
                    #subprocess.Popen(pylib_root_cmd + 'helpers\\download_images.py' + ' -i' + ' ' + proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '.csv', shell=True).wait()
                    # + ' [-d \'../../../../all_images\']')

                    # make raw images folder
                    imgdir = proj_dir + "/data/other/genus_download_records/iNat_images-" + genus + "-raw_images"
                    if not os.path.exists(imgdir):
                        os.mkdir(imgdir)
                    fileout = proj_dir + "/data/other/genus_download_records/" + genus + '-download_log.csv'
                    downloadImages(img_records=proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '.csv',imgdir=imgdir,fileout=fileout)
                    print("Finished downloading images for genus " + genus)

                    dirname = proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + "-raw_images"

                    # add INAT tag to id
                    # used to add tags for multiple images of same obs
                    names = os.listdir(dirname)
                    new_names = []
                    counter = 1
                    prev_id = ""
                    for i, name in enumerate(names):
                        id = name.split('_')[0]
                        if id != prev_id:
                            counter = 1
                        newname = id #+ "-" + str(counter)
                        os.rename(dirname + "/" + name,dirname + "/INAT-" + newname)
                        #new_names.append(id + "_" + str(counter))
                        counter += 1
                    print("Finished fixing same observation image names")

                    print("Started renaming images as JPGs, then moving them to all_images")
                    # rename images as JPGs and move
                    dirname = proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + "-raw_images"
                    for i, filename in enumerate(os.listdir(dirname)):
                        file = Path(dirname + "/" + filename + ".jpg")
                        if not file.exists():
                            os.rename(dirname + "/" + filename, dirname + "/" + filename + ".jpg")
                        shutil.move(dirname + "/" + filename + ".jpg", proj_dir + "/data/all_images/" + filename + ".jpg")
                    print("Started renaming images as JPGs, then moving them to all_images")

                    # if there are more chunks left, wait for an hour before doing the next chunk
                    if c < len(os.listdir(proj_dir + '/data/other/genus_download_records/iNat_images-' + genus + '-records_split')) - 1:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print("Waiting one hour starting at " + current_time)
                        time.sleep(3600)
