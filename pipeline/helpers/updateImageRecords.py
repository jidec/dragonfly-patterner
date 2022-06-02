import os

# may not go this route
def updateImageRecords(proj_dir):
    """
        Update records using downloaded image names
        Key reason for this is the fact that often multiple images correspond to the same record
        Thus, new rows need to be added to the records dataframe when new images are downloaded
        :param str proj_root: the location of the project folder i.e. dragonfly-patterner containing an /R/src/preprocessiNat.R script to source from
    """

    # get all image files
    files = os.listdir(proj_dir + "/data/all_images")