from getFilterImageIDs import getFilterImageIDs
from inferSegments import inferSegments

# get image IDs classified as not being bad, not already segmented, and not already segmented in training
image_ids = getFilterImageIDs(infer_fields=["class","class","class","has_segment_infer"],infer_values=["dorsal","lateral","dorsolateral",-1.0],
                              train_fields=["has_segment"],train_values=[-1.0])

# infer segments
inferSegments(image_ids=image_ids,model_location='../../data/ml_models/segmenter.pt',image_size=344,show=False)
