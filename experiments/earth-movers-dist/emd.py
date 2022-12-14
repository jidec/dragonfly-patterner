from pyemd import emd_samples
import cv2

img = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-130715589-1_pattern.png") # dorsal gomphid
img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-13184743-8_pattern.png") # ok gomphid 2
img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-26674196-2_pattern.png") # ok gomphid
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-11921484-1_pattern.png") # fucky
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-28006702-1_pattern.png") # fucked gomphid
#img2 = cv2.imread("E:\dragonfly-patterner\data\patterns\gomph_grouped_5000\INAT-5635184-2_pattern.png") # mega fucked

img2 = cv2.resize(img2, (img.shape[1],img.shape[0]))

print(emd_samples(img,img2))