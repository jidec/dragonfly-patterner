# for each species represented in
# blendSpecies(species_list)
from procrustesComposite import procrustesComposite

image_ids = [str(x) for x in [1,2,3,6,9]]

procrustesComposite(image_ids,show=True)