from extractHoneSegments import extractHoneSegments

# get image IDs with segments
image_ids = ["AW-focol0769-h","AW-casent0612213-h","AW-casent0187063-h","AW-casent0101436-h","AW-csironc0066-h","AW-casent0609019-h","AW-focol02971-h","AW-focol0769-h"]

# extract and refine segments
rgba_imgs_masks_names = extractHoneSegments(image_ids,bound=True,remove_islands=True,erode=True,erode_kernel_size=3,adj_to_background_grey=True,
                                        write=False,show=True,proj_dir="E:/ant-patterner")