import os
import requests
import subprocess
from get_inat_records2 import getiNatRecords

#shell_cmd = "get_inat_records.py Anax"
#subprocess.Popen(shell_cmd, shell=True).wait()
#os.system('get_inat_records.py') #+ ' [-r]')

getiNatRecords(genus="Dythemis",proj_dir="../../../..")