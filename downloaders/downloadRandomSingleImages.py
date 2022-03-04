import requests
import json
import pandas as pd
n = 400
data = pd.read_csv('../data/inat_odonata_usa.csv',sep='\t')
ids = data.sample(n)['catalogNumber']

for i in ids:
    try:
        link = 'https://api.inaturalist.org/v1/observations/' + str(i)
        x = requests.get(link)
        obs = json.loads(x.text)

        # parse down to image url
        result = obs.get("results")
        result = result[0]
        result.keys()
        result = result.get('observation_photos')
        result = result[0]
        result = result.get('photo')
        result = result.get('url')

        # replace square with original to get full size
        result = result.replace("square", "original")
        img = requests.get(result).content
        file = open("../data/random_images/" + str(i) + ".jpg", "wb")

        file.write(img)
        file.close()
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        print('Decoding JSON has failed at ' + link)
