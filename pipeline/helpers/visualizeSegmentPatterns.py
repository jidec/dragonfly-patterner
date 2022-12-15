from showImages import showImages
import cv2

def visualizeSegmentPatterns(ids, pattern_dirs=["","grouped"],proj_dir="../.."):
    for id in ids:
        seg = cv2.imread(proj_dir + "/data/segments/" + id + "_segment.png")
        pats = []
        pat_path = proj_dir + "/data/patterns/"
        for dir in pattern_dirs:
            pats.append(cv2.imread(pat_path + dir + "/" + id + "_pattern.png"))

        showImages(True, [seg] + pats, ["segment"] + pattern_dirs)