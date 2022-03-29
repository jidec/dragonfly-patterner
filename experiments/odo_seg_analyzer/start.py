from extractSegment import extractSegment
from discretizeIndv import discretizeIndv
from alignDiscretized import alignDiscretized
from rotateToVertical import rotateToVertical

img_name = "56568289_564"
img_name = "4155610_198"
#rgba_img_and_mask = extractSegment(img_name = img_name, img_dir = "images", mask_dir="images",show=True)
#discretizeIndv(rgba_img_and_mask=rgba_img_and_mask, img_name = img_name, by_contours=True, cluster_model="kmeans", out_dir="image", show=True)
#rotateToVertical(img_name,img_dir="images",out_dir="images",show=True)
alignDiscretized(ref_img_name="56568289_564",ref_img_dir="images", img_name="4155610_198",img_dir="images",out_dir="images",show=True)