from downloadiNatGenusImages import downloadiNatGenusImages

#aesh_genera_indices = [19, 21, 31, 43, 44, 49, 52, 54, 58, 61, 69, 87, 88]
#aesh_genera_indices = [31, 43, 44, 49, 52, 54, 58, 61, 69, 87, 88]
gomph_indices = [50,55,  9, 59, 23, 24, 28, 83, 18, 14, 47, 42, 63, 57, 91, 62, 73, 85]

for i in gomph_indices:
    downloadiNatGenusImages(start_index=i,end_index=i+1)