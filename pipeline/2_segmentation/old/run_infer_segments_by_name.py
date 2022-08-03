from inferSegments import inferSegments

image_ids = ["INATRANDOM-746982","INATRANDOM-1070520","INATRANDOM-2778053"]

# infer segments
inferSegments(image_ids=image_ids,model_location='../../data/ml_models/segmenter.pt',image_size=344,show=True)
