from preprocessRecords import preprocessRecords
from writeiNatGenusList import writeiNatGenusList

# for dragonfly-patterner, download Odonata research-grade iNat observations from USA
# https://www.gbif.org/occurrence/search?country=US&dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=789
# place in root of data folder and rename as <preferred_dataset_name>_records.csv
# _records is a needed identifier for downstream functions

# add imageID and dataSource columns to record files
#preprocessRecords(csv_names=["inatdragonflyusa_records"],id_cols=["catalogNumber"],csv_seps=['\t'])
#preprocessRecords(csv_names=["inatdragonflyusa_records"],id_cols=["catalogNumber"],csv_seps=['\t'])

# write genus list for iNat downloading
writeiNatGenusList(inat_csv_name="inatdragonflyusa_records")