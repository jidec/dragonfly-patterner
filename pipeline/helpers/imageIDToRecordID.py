
# quick helper to split image ids to record ids
def imageIDToRecordID(image_ids):
    record_ids = []
    for id in image_ids:
        id = id.split("-")[0] + "-" + id.split("-")[1]
        record_ids.append(id)
    return record_ids