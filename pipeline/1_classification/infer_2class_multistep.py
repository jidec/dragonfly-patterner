from inferImages import inferImages

## Infer dorsal or non dorsal, then dorsal good or dorsal perfect
## also infer lateral or non lateral, then lateral good or lateral perfect

image_dir = '../../data/random_images'

# infer dorsal or non dorsal
dorsal_non_inference = inferImages(image_dir=image_dir, model_location='../experiments/odo_view_classifier/dorsal_non/dorsal_non_model.pt',
                                   image_size=344,show=True)
# infer dorsal or dorsal perfect
dorsal_perfect_inference = inferImages(image_dir=image_dir, model_location='../experiments/odo_view_classifier/dorsal_non/dorsal_perfect_model.pt',
                                       image_size=344, show=True)

#----------
# infer lateral or non lateral
lateral_non_inference = inferImages(image_dir=image_dir, model_location='../experiments/odo_view_classifier/lateral_non/lateral_non_model.pt',
                                   image_size=344,show=True)
# infer lateral or lateral perfect
lateral_perfect_inference = inferImages(image_dir=image_dir, model_location='../experiments/odo_view_classifier/lateral_non/lateral_perfect_model.pt',
                                       image_size=344, show=True)
# combine inferences to infer lateral, non-lateral, or lateral perfect
lateral_non_perfect_inference = lateral_non_inference
for i in range(1,len(lateral_non_inference)):
    if (lateral_non_perfect_inference[i] == "lateral") & (lateral_perfect_inference[i] == "perfect"):
        lateral_non_perfect_inference[i] = "perfect"

# check number of matches to test