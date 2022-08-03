
import requests
import csv
import string
import random
import time, datetime
import os.path
from argparse import ArgumentParser

base_url = 'https://api.inaturalist.org/v1/'
print("Starting")

def getTaxonID(taxon_name):
    print("Getting taxon id")
    if len(taxon_name.split(' ')) == 2:
        rank = 'species'
    else:
        rank = 'genus'

    params = {'rank': rank, 'q': taxon_name}
    resp = requests.get(base_url + 'taxa', params=params)
    res = resp.json()

    match_cnt = 0
    taxon_id = None
    for t_info in res['results']:
        if t_info['name'] == taxon_name:
            taxon_id = t_info['id']
            match_cnt += 1

    if match_cnt == 0:
        raise Exception('Could not find the {0} "{1}".\n'.format(
            rank, taxon_name
        ))
    elif match_cnt > 1:
        raise Exception(
            'More than one {0} name match for "{1}".\n'.format(rank, taxon_name
        ))

    return taxon_id

def getRecCnt(base_params):
    print("Getting record count")
    resp = requests.get(base_url + 'observations', params=base_params)
    res = resp.json()

    return int(res['total_results'])

def getControlledVocab():
    print("Getting controlled vocab")
    """
    Retrieves the controlled vocabulary used for annotations.
    """
    resp = requests.get(base_url + 'controlled_terms')
    res = resp.json()

    vocab = {}

    for result in res['results']:
        vocab[result['id']] = result['label']
        for val in result['values']:
            vocab[val['id']] = val['label']

    return vocab
    
def getRecords(base_params, writer, prev_obs_ids, full_size, vocab=None):
    print("Records")
    if vocab is None:
        vocab = getControlledVocab()

    params = base_params.copy()
    params['per_page'] = 200

    # Keep track of which observation IDs we've seen to ensure there are no
    # duplicates.
    obs_ids = set()

    # Initialize the stack of time intervals.
    end_ts = time.time()
    start_ts = time.mktime(time.strptime(
        'Jan 01 2008 00:00:00', '%b %d %Y %H:%M:%S'
    ))
    time_stack = [(start_ts, end_ts)]

    TF_STR = '%d %b %Y'

    # Remove time intervals from the stack until the stack is empty, splitting
    # intervals as needed to get the total number of records per interval below
    # 4,000, then downloading all records for each usable interval.
    while len(time_stack) > 0:
        start_ts, end_ts = time_stack.pop()

        s_time_str = datetime.datetime.fromtimestamp(start_ts).isoformat()
        e_time_str = datetime.datetime.fromtimestamp(end_ts).isoformat()
        params['created_d1'] = s_time_str
        params['created_d2'] = e_time_str

        rec_cnt = getRecCnt(params)

        if rec_cnt > 4000:
            print('Splitting interval with {0:,} records ({1} - {2})...'.format(
                rec_cnt, time.strftime(TF_STR, time.localtime(start_ts)),
                time.strftime(TF_STR, time.localtime(end_ts))
            ))
            mid_ts = ((end_ts - start_ts) / 2) + start_ts
            time_stack.append((mid_ts, end_ts))
            time_stack.append((start_ts, mid_ts))
        else:
            print('Getting {0:,} records for interval {1} - {2}...'.format(
                rec_cnt, time.strftime(TF_STR, time.localtime(start_ts)),
                time.strftime(TF_STR, time.localtime(end_ts))
            ))
            retrieveAllRecords(
                params, writer, prev_obs_ids, full_size, obs_ids, vocab
            )

def retrieveAllRecords(
    base_params, writer, prev_obs_ids, full_size, obs_ids, vocab
):
    FNCHARS = string.digits + string.ascii_letters

    params = base_params.copy()
    more_records = True
    record_cnt = 0
    page = 1

    print('Retrieving records...')

    while more_records:
        params['page'] = page
        resp = requests.get(base_url + 'observations', params=params)
        res = resp.json()
        #print(res)
        if len(res['results']) < params['per_page']:
            more_records = False

        row_out = {}
        for rec in res['results']:
            record_cnt += 1

            obs_id = rec['id']
            if obs_id in prev_obs_ids:
                continue

            if obs_id in obs_ids:
                raise Exception(
                    'Repeat observation encountered: {0}'.format(obs_id)
                )
            else:
                obs_ids.add(obs_id)


            img_list = rec['photos']

            # LOOP EDITED
            # if has an image
            if len(img_list) > 0:
                # for each image in img_list
                for i in range(0,len(img_list)-1):
                    img = img_list[i]
                    img_num = i + 1
                    if (
                        img['url'] is not None and
                        img['original_dimensions'] is not None
                    ):
                        #img_fname = ''.join([
                        #    random.choice(FNCHARS) for cnt in range(16)
                        #]) + '.jpg'
                        img_fname = '' + str(obs_id) + '-' + str(img_num) + '.jpg' #str(random.randint(0, 1000)) + '.jpg' #hacky solution here OLD: .join(obs_id)
                        if full_size:
                            img_url = img['url'].replace('/square.', '/original.')
                        else:
                            img_url = img['url'].replace('/square.', '/large.')

                        # Get the annotations.
                        annot_list = []
                        for annot in rec['annotations']:
                            attr = annot['controlled_attribute_id']
                            value = annot['controlled_value_id']
                            annot_list.append(
                                '{0}:{1}'.format(vocab[attr], vocab[value])
                            )

                        if rec['location'] is not None:
                            latlong = rec['location'].split(',')
                        else:
                            latlong = ('', '')

                        row_out['obs_id'] = obs_id
                        row_out['usr_id'] = rec['user']['id']
                        row_out['date'] = rec['observed_on']
                        row_out['latitude'] = latlong[0]
                        row_out['longitude'] = latlong[1]
                        row_out['taxon'] = rec['taxon']['name']
                        row_out['img_cnt'] = len(img_list)
                        row_out['img_id'] = img['id']
                        row_out['file_name'] = img_fname
                        row_out['img_url'] = img_url
                        row_out['width'] = img['original_dimensions']['width']
                        row_out['height'] = img['original_dimensions']['height']
                        row_out['license'] = img['license_code']
                        row_out['annotations'] = ','.join(annot_list)
                        row_out['tags'] = ','.join(rec['tags'])
                        row_out['download_time'] = time.strftime(
                            '%Y%b%d-%H:%M:%S', time.localtime()
                        )
                        writer.writerow(row_out)

        print('  {0:,} records processed...'.format(record_cnt))
        page += 1

    print('done.')

def readExtantObsIds(fpath):
    obs_ids = set()

    with open(fpath, encoding="utf-8") as fin: #added encoding="utf-8"
        reader = csv.DictReader(fin)
        for row in reader:
            obs_ids.add(int(row['obs_id']))

    return obs_ids


argp = ArgumentParser(
    description='Downloads iNaturalist observation records for a given genus '
    'or species.  Generates a CSV file of records that can be used with '
    'download.py to retrieve the actual image files.'
)
argp.add_argument(
    '-o', '--output_file', type=str, required=False, default=None,
    help='An output CSV file name (default: "iNat_images-TAXON_NAME.csv").'
)
argp.add_argument(
    '-r', '--research_only', action='store_true',
    help='If this flag is set, only research-grade observations will be used.'
)
argp.add_argument(
    '-c', '--counts_only', action='store_true',
    help='If this flag is set, only report record counts; do not actually '
    'download all records.'
)
argp.add_argument(
    '-f', '--full_size', action='store_true',
    help='If this flag is set, full-size images will be downloaded.'
)
argp.add_argument(
    '-u', '--update', action='store_true',
    help='Run in update mode, which means that new iNaturalist records will be '
    'added to the output file without overwriting records already in the '
    'output file.'
)
argp.add_argument(
    'taxon', type=str, help='A genus or species name.'
)

args = argp.parse_args()
args.taxon = "Dythemis"
try:
    taxon_id = getTaxonID(args.taxon)
except Exception as err:
    exit('\nERROR: ' + str(err))

base_params = {'taxon_id': taxon_id}
if args.research_only:
    print("Research only")
    base_params['quality_grade'] = 'research'

if args.counts_only:
    cnt = getRecCnt(base_params)

    msg = '\n{0:,} records available for {1}'.format(cnt, args.taxon)
    if args.research_only:
        msg += ' (research-grade only).\n'
    else:
        msg += '.\n'

    print(msg)
else:
    ofpath = args.output_file
    if ofpath is None:
        #ofpath = 'helpers/genus_image_records/iNat_images-' + args.taxon.replace(' ', '_') + '.csv' #EDITED
        # add param for proj_root
        ofpath = '../../data/other/genus_download_records/iNat_images-' + args.taxon.replace(' ', '_') + '.csv'  # EDITED
    of_exists = os.path.exists(ofpath)
    prev_obs_ids = {}
    if of_exists:
        if not(args.update):
            exit('\nERROR: The output file, {0}, already exists.\n'.format(
                ofpath
            ))
        else:
            prev_obs_ids = readExtantObsIds(ofpath)

    fout = open(ofpath, 'a', encoding='utf-8')
    writer = csv.DictWriter(fout, [
        'obs_id', 'usr_id', 'date', 'latitude', 'longitude', 'taxon',
        'img_cnt', 'img_id', 'file_name', 'img_url', 'width', 'height',
        'license', 'annotations', 'tags', 'download_time'
    ])
    if not(of_exists):
        writer.writeheader()

    getRecords(base_params, writer, prev_obs_ids, args.full_size)

