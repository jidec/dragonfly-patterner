from inferImages import inferImages

## Infer dorsal, lateral, or bad, then infer perfect dorsals and perfect laterals

# set dir to infer images from
image_dir = '../../data/random_images'

# infer dorsal, lateral, or bad
dorsal_lateral_bad_inference = inferImages(image_dir=image_dir,
                                   model_location='../experiments/odo_view_classifier/dorsal_lateral_bad/dorsal_lateral_bad_model.pt',
                                   image_size=344,show=True)
# infer dorsal or dorsal perfect
dorsal_perfect_inference = inferImages(image_dir=image_dir, model_location='../experiments/odo_view_classifier/dorsal_perfect/dorsal_perfect_model.pt',
                                       image_size=344, show=True)
# combine inferences to infer dorsal, non-dorsal, or dorsal perfect
for i in range(1,len(dorsal_lateral_bad_inference)):
    if (dorsal_lateral_bad_inference[i] == "dorsal") & (dorsal_perfect_inference[i] == "perfect"):
        dorsal_lateral_bad_inference[i] = "dorsal_perfect"

# infer lateral or lateral perfect
lateral_perfect_inference = inferImages(image_dir=image_dir, model_location='../experiments/odo_view_classifier/lateral_perfect/lateral_perfect_model.pt',
                                       image_size=344, show=True)
# combine inferences to infer dorsal, non-dorsal, or dorsal perfect
for i in range(1,len(dorsal_lateral_bad_inference)):
    if (dorsal_lateral_bad_inference[i] == "lateral") & (lateral_perfect_inference[i] == "perfect"):
        dorsal_lateral_bad_inference[i] = "lateral_perfect"
