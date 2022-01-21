from pyinaturalist import *
import json
import csv
import pandas as pd

# really bad non vectorized way of doing it but I can't think of a better one

# for id in inat_odonata_usa.csv
id = 104479410
df = pd.DataFrame.from_dict(get_observation(id), orient='index')
df = df.loc["annotations"]
print(df)