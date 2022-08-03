
# Downloads images from a set of iDigBio search results, using random names for
# the downloaded files, and generates a CSV file mapping downloaded file names
# to source URIs, iDigBio core IDs, and scientific names.


import os.path
import csv
from argparse import ArgumentParser
import time
from imgfiles import mtDownload, ImageDownloadWorkerResult


def getArgParser():
    argp = ArgumentParser(
        description='Downloads images from records in a compatible CSV file.  '
        'All downloaded image files will be converted to JPEG format, if '
        'needed.'
    )
    argp.add_argument(
        '-i', '--img_records', type=str, required=True, help='The path to an '
        'input CSV file that includes columns "file_name" and "img_url".'
    )
    argp.add_argument(
        '-o', '--fileout', type=str, required=False,
        default=None, help='The path to an output file for logging download '
        'information (default: "IMG_RECORDS_FILE-download_log.csv").'
    )
    argp.add_argument(
        '-d', '--imgdir', type=str, required=False, default=None,
        help='The path to a directory in which to place the downloaded images '
        '(default: "IMG_RECORDS_FILE-raw_images"). The directory will be '
        'created if it does not already exist.'
    )
    argp.add_argument(
        '-t', '--timeout', type=int, required=False, default=20, help='The '
        'number of seconds to wait before HTTP connect or read timeout '
        'failures.'
    )
    argp.add_argument(
        '-n', '--no_skip', action='store_true', help='Do not skip images that '
        'already exist in the download folder (i.e., redownload extant images).'
    )
    argp.add_argument(
        '-p', '--no_append', action='store_true', help='Do not append new '
        'results to the output file if it already exists.'
    )
    argp.add_argument(
        '-r', '--threads', type=int, required=False, default=20, help='The '
        'maximum number of threads to use for concurrent downloads.'
    )

    return argp

def checkPaths(args):
    if not(os.path.exists(args.img_records)):
        raise IOError(
            'The input image records file, "{0}", could not be '
            'found.'.format(args.img_records)
        )

    if args.fileout is None:
        args.fileout = (
            os.path.splitext(args.img_records)[0] + '-download_log.csv'
        )
    if os.path.exists(args.fileout) and args.no_append:
        raise IOError(
            'The output file, "{0}", already exists.'.format(
                args.fileout
            )
        )

    if args.imgdir is None:
        args.imgdir = os.path.splitext(args.img_records)[0] + '-raw_images'
    if os.path.exists(args.imgdir) and not(os.path.isdir(args.imgdir)):
        raise IOError(
            'A file with the same name as the output image directory, '
            '"{0}", already exists.'.format(args.imgdir)
        )

    if not(os.path.exists(args.imgdir)):
        os.mkdir(args.imgdir)

def getDownloadRequests(fpath, img_dir, skip_existing):
    """
    Generates a list of download request (file_name, URI) tuples.
    """
    downloads = []
    with open(fpath, encoding="utf-8") as fin:
        reader = csv.DictReader(fin)

        for line in reader:
            imguri = line['img_url']
            fname = line['file_name']
            tmp_fname = os.path.splitext(fname)[0]
            if os.path.exists(os.path.join(img_dir, fname)):
                if skip_existing:
                    print('Skipping extant image {0}...'.format(fname))
                else:
                    downloads.append((tmp_fname, imguri))
            else:
                downloads.append((tmp_fname, imguri))

    return downloads


argp = getArgParser()
#args = argp.parse_args()
args, unknown = argp.parse_known_args()

try:
    checkPaths(args)
except Exception as err:
    exit('\nERROR: {0}\n'.format(err))

downloads = getDownloadRequests(
    args.img_records, args.imgdir, not(args.no_skip)
)
if len(downloads) == 0:
    exit()

# Generate the file name for the failure log.
logfn = 'fail_log-{0}.csv'.format(
    time.strftime('%Y%b%d-%H:%M:%S', time.localtime())
)

# Process the download requests.
with open(args.fileout, 'a', encoding="utf-8") as fout: #, open(logfn, 'w') as logf:
    writer = csv.DictWriter(
        fout, ['file_name', 'file_path', 'imgsize', 'bytes', 'img_url', 'time']
    )
    writer.writeheader()
    outrow = {}

    #faillog = csv.DictWriter(
    #    logf, ['file_name', 'img_url', 'time', 'reason']
    #)
    #faillog.writeheader()
    logrow = {}

    for result in mtDownload(
        downloads, args.imgdir, args.timeout, args.threads
    ):
        if result.result == ImageDownloadWorkerResult.SUCCESS:
            outrow['file_name'] = result.identifier
            outrow['file_path'] = result.localpath
            outrow['img_url'] = result.uri
            #outrow['imgsize'] = getImageSize(result.localpath)
            #outrow['bytes'] = os.stat(result.localpath).st_size
            outrow['time'] = result.timestamp
            writer.writerow(outrow)
        elif result.result == ImageDownloadWorkerResult.DOWNLOAD_FAIL:
            logrow['file_name'] = result.identifier
            logrow['img_url'] = result.uri
            logrow['time'] = result.timestamp
            logrow['reason'] = result.fail_reason
            #faillog.writerow(logrow)

