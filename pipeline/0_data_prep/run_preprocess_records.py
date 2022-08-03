from mergePreprocessRecords import mergePreprocessRecords
from writeiNatGenusList import writeiNatGenusList

# for dragonfly-patterner, download Odonata research-grade iNat observations from USA
# https://www.gbif.org/occurrence/search?country=US&dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&taxon_key=789
# place in root of data folder and rename as <preferred_dataset_name>_records.csv
# _records is a needed identifier for downstream functions

mergePreprocessRecords(raw_records_csv_names=["inatdragonflyusa","odonatacentral"],
                       id_cols=["catalogNumber","OC #"],id_prefixes=["INAT","OC"],csv_seps=['\t',','])

writeiNatGenusList(inat_csv_name="inatdragonflyusa_records")