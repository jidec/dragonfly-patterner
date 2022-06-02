from preprocessRecords import preprocessRecords
from writeiNatGenusList import writeiNatGenusList
from getFilterImageIDs import getFilterImageIDs

# for dragonfly-patterner, download Odonata research-grade iNat observations from USA
# https://www.gbif.org/occurrence/search?country=US&dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=789
# place in root of data folder and rename as <preferred_dataset_name>_records.csv
# _records is a needed identifier for downstream functions

# run preprocess function
# preprocessRecords(csv_names=["inatdragonflyusa_records"],id_cols=["catalogNumber"],csv_seps=['\t'])

# create genus list from iNat data for downloaders to use, gets placed in downloaders folder in pipeline
# writeiNatGenusList(inat_csv_name="inatdragonflyusa_records")

getFilterImageIDs()