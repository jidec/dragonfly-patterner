import cv2
import pandas as pd
import numpy as np
from pyemd import emd_samples
from skimage.metrics import structural_similarity
from os.path import exists
from imageIDToRecordID import imageIDToRecordID
from itertools import compress

def inferQualityMetrics(image_ids,metrics=["emd_pat", "emd_seg",
                                                   "mse_seg","mse_pat",
                                                   "struct_diff_seg","struct_sim_pat",
                                                    "symmetry_pat", "symmetry_seg",
                                                   "light_glared_seg","dim_diff"],
                        quantiles = [0.9],
                        group_records_col="genus",print_steps=True,proj_dir="../.."):
    segs = []
    pats = []
    if print_steps: print("Reading segments and patterns")
    for id in image_ids:
        segs.append(cv2.imread(proj_dir + "/data/segments/" + id + "_segment.png"))
        pats.append(cv2.imread(proj_dir + "/data/patterns/" + id + "_pattern.png"))

    # if not grouping by records col
    if group_records_col == None:
        if print_steps: print("Grouping provided IDs together")
        df = getGroupMetricDF(metrics,group_pats=pats,group_segs=segs)
        df['imageID'] = image_ids

    # otherwise group by records col
    else:
        if print_steps: print("Grouping IDs by column " + group_records_col)

        # convert given ids to record ids
        img_record_ids = imageIDToRecordID(image_ids)

        # load records and keep those with matching ids
        records = pd.read_csv(proj_dir + "/data/records.csv")
        records = records[records["recordID"].isin(img_record_ids)]

        # get unique groups
        unique_groups = pd.unique(records[group_records_col])

        # create data frame to concat group inferences to
        df = pd.DataFrame()

        # for each unique group (i.e. species)
        for g in unique_groups:
            # get records for the group (records where the group col matches)
            group_records = records[records[group_records_col] == g]
            # get a list of ids for the group
            group_ids = list(set(group_records['recordID'].tolist()))

            # get indices in given ids matching group
            group_bool_indices = [True if id in group_ids else False for id in img_record_ids]
            #group_indices = list(compress(range(len(group_bool_indices)), group_bool_indices))

            # get segs and pats for the group using the indices
            group_segs = list(compress(segs, group_bool_indices))
            group_pats = list(compress(pats, group_bool_indices))

            if print_steps: print("Started getting metrics for group " + g)

            group_df = getGroupMetricDF(metrics, group_pats=group_pats,group_segs=group_segs)
            group_df['imageID'] = image_ids[group_bool_indices]
            df = pd.concat([df,group_df])

    # metric names is all columns except last (which is imageID)
    metric_names = df.columns[:-1]
    for colname in metric_names:
        df[colname + "_percentiles"] = df[colname].rank(pct=True)
        for q in quantiles:
            df[colname + "_quantile" + str(q)] = df[colname + "_percentiles"] > q

    if exists(proj_dir + "/data/inferences.csv"):
        current_infers = pd.read_csv(proj_dir + "/data/inferences.csv")
        inferences = pd.merge(current_infers,df,how="outer",on="imageID")


    inferences.to_csv(proj_dir + "/data/inferences.csv")
    if print_steps: print("Saved all inferences")

def getGroupMetricDF(metrics,group_pats,group_segs):
    metric_names = []
    metric_values = []

    if "emd_pat" in metrics:
        metric_values.append(metricPairwiseRowMean(group_pats, emd_samples))
        metric_names.append("emd_pat")

    if "mse_pat" in metrics:
        metric_values.append(metricPairwiseRowMean(group_pats, emd_samples))
        metric_names.append("mse_pat")

    if "struct_diff_pat" in metrics:
        metric_values.append(metricPairwiseRowMean(group_pats, structDiff))
        metric_names.append("struct_diff_pat")

    if "symmetry_pat" in metrics:
        metric_values.append(mseSymmetry(group_pats))
        metric_names.append("symmetry_pat")

    if "light_glared_seg" in metrics:
        metric_values.append(lightnessGlared(group_segs))
        metric_names.append("light_glared_seg")

    if "dim_diff" in metrics:
        metric_values.append(metricPairwiseRowMean(group_segs, dimDiffH))
        metric_names.append("dim_diff_h")
        metric_values.append(metricPairwiseRowMean(group_segs, dimDiffW))
        metric_names.append("dim_diff_w")

    df = pd.DataFrame(metric_values).transpose()
    df.columns = metric_names
    return df

def metricPairwiseRowMean(imgs, func, resize=False):
    mat = np.empty((len(imgs), len(imgs)))
    #print(mat)
    for i, img in enumerate(imgs):
        for i2, img2 in enumerate(imgs):
            # fix dims if necessary
            img2 = cv2.resize(img2,(np.shape(img)[1],np.shape(img)[0]))
            mat[i, i2] = func(img,img2)
            #print(i,i2)
        print(i)

    rowmean = np.mean(mat, axis=1)
    return(rowmean)

def meanSquaredError(img, img2, greyscale=True):
    if greyscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    diff = cv2.subtract(img, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse

def dimDiff(img, img2, axis):
    # dim diff between two images
    diff = abs(np.shape(img)[axis] - np.shape(img2)[axis])
    return diff

def dimDiffH(img,img2): return(dimDiff(img,img2,axis=0))
def dimDiffW(img,img2): return(dimDiff(img,img2,axis=1))

def structDiff(img, img2, greyscale=True):
    if greyscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = structural_similarity(img, img2, full=True)

    return score

def mseSymmetry(imgs, greyscale=True):
    syms = []
    for img in imgs:
        height = img.shape[0]
        width = img.shape[1]

        # cut the image in half
        width_cutoff = width // 2
        img_left = img[:, :width_cutoff]
        img_right = img[:, width_cutoff:]

        img_right = cv2.flip(img_right, 1)

        # resize right to left just in case
        img_right = cv2.resize(img_right, (img_left.shape[1], img_left.shape[0]))

        syms.append(meanSquaredError(img_left, img_right))
    return syms

# glared if at least 10% of pixels have raw lightness values greater than 0.9
def lightnessGlared(imgs, lightness_cutoff=0.9, percent_past_cutoff=0.1):
    is_glared_vect = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        is_glared = (np.count_nonzero(img > lightness_cutoff) / (np.shape(img)[1] * np.shape(img)[0])) > percent_past_cutoff
        is_glared_vect.append(is_glared)
    return(is_glared_vect)


#l < - (img[,, , 1] + img[, , , 2] + img[, , , 3]) / 3
#is_glared < - (sum(l > lightness_cutoff) / length(l)) > percent_past_cutoff
#is_glared_vect < - c(is_glared_vect, is_glared)
#}
# PAT SEG LOAD SNIPPETS
# get all segments from segments folder and subfolders
#seg_paths = []
#for dir, _, _ in os.walk(proj_dir + "/data/segments"):
#    seg_paths.extend(glob(os.path.join(dir, "*.png")))
#segs = [cv2.imread(p) for p in seg_paths]

#if image_ids is None:
#    seg_paths = glob.glob(proj_dir + "/data/segments/*.png")
#    pat_paths = glob.glob(proj_dir + "/data/patterns/*.png")
#else:
    # seg paths are all paths in the seg folder IF contains an ID in the list
#    seg_paths = [p for p in glob.glob(proj_dir + "/data/segments/*.png") if any(substring in p for substring in image_ids)]
#    pat_paths = [p for p in glob.glob(proj_dir + "/data/patterns/*.png") if any(substring in p for substring in image_ids)]

# sort paths so that they can be matched to records IDs
#seg_paths.sort()
#pat_paths.sort()

# load segs and pats from paths
#segs = [cv2.imread(file) for file in seg_paths]
#pats = [cv2.imread(file) for file in pat_paths]