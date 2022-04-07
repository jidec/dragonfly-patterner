import requests
import json
import pandas as pd

def downloadRandomSingleImages(n,pylib_root=''):
    data = pd.read_csv(pylib_root + '/../../../data/inat_odonata_usa.csv',sep=',')
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
            file = open(pylib_root + "/../../../data/all_images/" + str(i) + ".jpg", "wb")

            file.write(img)
            file.close()
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed at ' + link)
