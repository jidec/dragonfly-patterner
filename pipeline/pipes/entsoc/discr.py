from colorDiscretize import colorDiscretize
from getFilterImageIDs import getFilterImageIDs
import random
import cv2

#ids = getFilterImageIDs(not_in_train_data=True,infer_fields=["bad_signifier"],infer_values=[0],
                        #proj_dir="../../..")
#ids = ids[0:100]
ids = ['INAT-16597140-4'] #'INAT-31693070-5']#, 'INAT-25972892-2','INAT-33735041-12','INAT-66122862-12','INAT-50009064-2', 'INAT-15787168-7', 'INAT-133070548-1','INAT-87913756-1','INAT-18022507-15', 'INAT-59943291-2', 'INAT-17272251-5', 'INAT-12956867-8', 'INAT-8065771-7', 'INAT-98818041-3', 'INAT-8541039-1', 'INAT-94257527-1', 'INAT-80324990-1', 'INAT-33094909-1', 'INAT-7971868-6', 'INAT-86292303-1', 'INAT-5276224-1', 'INAT-28812623-1', 'INAT-30782356-11', 'INAT-7201668-1', 'INAT-137699842-1', 'INAT-137699827-3', 'INAT-47183338-1', 'INAT-93438843-2', 'INAT-13436921-10', 'INAT-14678965-10', 'INAT-33340999-3', 'INAT-32660811-1', 'INAT-32261357-1', 'INAT-22059144-4', 'INAT-8437742-6','INAT-94476418-1', 'INAT-93879700-1', 'INAT-59519976-3', 'INAT-132365449-1', 'INAT-104843569-2', 'INAT-16110236-1', 'INAT-12967145-1', 'INAT-8291392-19', 'INAT-33102662-2', 'INAT-7936387-6', 'INAT-118420524-2', 'INAT-31147906-1', 'INAT-17891891-11', 'INAT-6738464-2', 'INAT-9785674-1', 'INAT-30123947-2', 'INAT-82855526-2', 'INAT-8180325-18', 'INAT-8598225-5', 'INAT-4138354-3', 'INAT-6805135-8', 'INAT-21118100-1', 'INAT-40922152-3', 'INAT-33735042-5', 'INAT-4171834-1', 'INAT-123139843-6', 'INAT-6228815-2', 'INAT-138175620-1', 'INAT-31688984-4', 'INAT-30213538-2', 'INAT-66064598-4', 'INAT-8180176-5', 'INAT-7936387-2', 'INAT-31456894-11', 'INAT-28708653-1', 'INAT-63351619-1', 'INAT-63127453-1', 'INAT-8598225-7', 'INAT-98938620-2', 'INAT-15216037-1', 'INAT-12974892-2', 'INAT-60361619-2', 'INAT-125770480-1', 'INAT-7385683-1', 'INAT-1808432-1', 'INAT-111881655-3', 'INAT-53218018-2', 'INAT-130674536-2', 'INAT-5338989-4', 'INAT-58802534-2', 'INAT-128430768-2', 'INAT-131029804-3', 'INAT-31457684-1', 'INAT-14676679-14', 'INAT-117795041-1', 'INAT-17272252-13', 'INAT-123184381-1', 'INAT-7970834-8', 'INAT-32427349-6', 'INAT-30061759-1', 'INAT-64419489-9', 'INAT-30402305-3', 'INAT-66363876-10', 'INAT-56947681-1']

#colorDiscretize(ids,by_contours=False, vert_resize=300,show=False,color_fun=cv2.COLOR_RGB2LAB,write_subfolder="lab/",nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,by_contours=False, vert_resize=400,show=False,color_fun=None,write_subfolder="rgb/",nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,by_contours=False, vert_resize=400,show=False,color_fun=cv2.COLOR_RGB2HSV,write_subfolder="rgb/",nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)

#colorDiscretize(ids,by_contours=False, vert_resize=300,show=False,color_fun=cv2.COLOR_RGB2LAB,write_subfolder="lab_scaled/",scale=True,nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,by_contours=False, vert_resize=400,show=False,color_fun=None,write_subfolder="rgb_scaled/",scale=True,nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,by_contours=False, vert_resize=400,show=False,color_fun=cv2.COLOR_RGB2HSV,write_subfolder="hsv_scaled/",scale=True,nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)

#colorDiscretize(ids,by_contours=False, vert_resize=300,show=False,color_fun=cv2.COLOR_RGB2LAB,write_subfolder="lab_scaled/",scale=True,nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,by_contours=False, vert_resize=400,show=False,color_fun=cv2.COLOR_RGB2HSV,write_subfolder="hsv_scaled_up/",scale=True,upweight_axis=0,nclusters=None,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,by_contours=False,vert_resize=400,show=False,color_fun=cv2.COLOR_RGB2HLS,write_subfolder="hls_sc_d/",scale=True,downweight_axis=1,cluster_positions=False,nclusters=3,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,by_contours=False,vert_resize=400,show=False,color_fun=cv2.COLOR_RGB2HLS,write_subfolder="hls_sc_d_pos/",scale=True,downweight_axis=3,cluster_positions=True,nclusters=3,cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
#colorDiscretize(ids,preclustered=False, write_subfolder="rgb",
#                colorspace=None, scale=False,downweight_axis=None,use_positions=False,proj_dir="../../..",print_details=True)
#colorDiscretize(ids,preclustered=False, write_subfolder="rgb_scale",
#                scale=True,downweight_axis=None,use_positions=False,proj_dir="../../..",print_details=True)
#colorDiscretize(ids,preclustered=False, write_subfolder="rgb_pos",
#                scale=False,downweight_axis=None,use_positions=True,proj_dir="../../..",print_details=True)
colorDiscretize(ids,preclustered=False, write_subfolder="",
                scale=False,downweight_axis=None,use_positions=False,proj_dir="../../..",print_details=True,show=True)
colorDiscretize(ids,preclustered=True, write_subfolder="grouped_scale",
                scale=True,downweight_axis=None,proj_dir="../../..",print_details=True)
#colorDiscretize(ids,preclustered=False, write_subfolder="lab",
#                colorspace="lab", scale=False,downweight_axis=None,use_positions=False,proj_dir="../../..",print_details=True)

#colorDiscretize(ids,preclustered=True,group_cluster_raw_ids=True,
#                colorspace=None, nclusters=5, scale=False,downweight_axis=None,use_positions=False,proj_dir="../../..",print_details=True)


#colorDiscretize(ids,cluster_2=True,group_cluster_raw_ids=True,vert_resize=250,show=False,nclusters=10, cluster_model="gaussian_mixture",proj_dir="../../..",print_details=True)
