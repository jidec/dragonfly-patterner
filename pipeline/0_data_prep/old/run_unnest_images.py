from unnestImagesAntWeb import unnestImagesAntWeb
from deleteFolders import deleteFolders
from deleteIfContains import deleteIfContains
from removeFromFilenames import removeFromFilenames

# functions to process a folder of raw AntWeb images (a specific and unusual case)
# unnestImagesAntWeb("E:/ant-patterner/data/new_images/antweb-images/antweb-images")

dir = "D:/ants/antweb-images"
#deleteFolders(dir)
#deleteIfContains(dir, "_2_")
#deleteIfContains(dir, "_3_")
removeFromFilenames(dir, "_1_med")

#sdeleteIfContains(direct_dir=dir,contains="h_segment",if_not_contains=True)

