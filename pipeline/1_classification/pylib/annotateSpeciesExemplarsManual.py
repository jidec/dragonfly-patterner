import pandas as pd
from getFilterImageIDs import getFilterImageIDs
import cv2
from showImages import showImages
from readchar import readkey

def manualAnnotateExemplars(exemplars_per_species_view, proj_root):
    # exemplars = pd.read_csv(proj_root + "/data/examplars.csv"
    records = pd.read_csv(proj_root + "/data/inatdragonflyusa_records.csv")
    species = pd.Series(records['species'].unique())

    for sp in species:
        for view in ['dorsal','lateral','dorsolateral']:
            ids = getFilterImageIDs(records_fields=["species"],records_values=[sp],infer_fields=["class"],infer_values=[view])

            new_examplars = []
            while len(new_examplars) < exemplars_per_species_view:
                # pop id then read and view image
                id = ids.pop(0)
                img = cv2.imread(proj_root + "/data/all_images/" + id + ".jpg", cv2.IMREAD_COLOR)
                showImages([img])

                # add answer to new examplars
                answer = readkey()
                if answer == 'y':
                    new_examplars.append([id,True])
                elif answer == 'f':
                    new_examplars.append([id, False])


    new_examplars = pd.DataFrame(new_examplars)
    new_examplars.to_csv(proj_root + "/data/exemplars.csv")


