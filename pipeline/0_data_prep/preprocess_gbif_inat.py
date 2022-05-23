from sourceRdefs import preprocessiNat, writeGenusList

# download USA odonata research-grade inat observations
# https://www.gbif.org/occurrence/search?country=US&dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=789
# place in root of data folder and rename as inat_data.csv

# run preprocess function
# Note - throws a replacement has 0 rows error if already preprocessed
preprocessiNat()

# create genus list from iNat data for downloaders to use, gets placed in downloaders folder in pipeline
writeGenusList()