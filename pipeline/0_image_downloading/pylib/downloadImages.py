
# Downloads images from a set of iDigBio search results, using random names for
# the downloaded files, and generates a CSV file mapping downloaded file names
# to source URIs, iDigBio core IDs, and scientific names.


import os.path
import csv
import time
from imgfiles import mtDownload, ImageDownloadWorkerResult

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
            downloads.append((tmp_fname, imguri))

    return downloads



def downloadImages(img_records,imgdir="../../data/other/",no_skip=False,fileout="IMG_RECORDS_FILE-download_log.csv",timeout=20,threads=20):
    downloads = getDownloadRequests(
        img_records, imgdir, not(no_skip)
    )
    if len(downloads) == 0:
        exit()

    # Generate the file name for the failure log.
    logfn = 'fail_log-{0}.csv'.format(
        time.strftime('%Y%b%d-%H:%M:%S', time.localtime())
    )

    # Process the download requests.
    with open(fileout, 'a', encoding="utf-8") as fout: #, open(logfn, 'w') as logf:
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
            downloads, imgdir, timeout, threads
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

