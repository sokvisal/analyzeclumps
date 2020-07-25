import os
import glob
from tqdm import tqdm
import subprocess
import shutil

def fit(path, tile):
    directories = [idnames for idnames in glob.glob('{}/a{}/_id-*'.format(path, tile)) ] #'../run/selectedgal/resolved_sfgs/at064/_id*' #'../run/at065/_id*'
    print ('Starting queue')
    wdir = os.getcwd()

    for i in directories[:]:

        if os.path.isdir(i+'/test_phot'):
            for dir in glob.glob('../analyzeclumps/sedfiles/*.param'):
                shutil.copyfile(dir, i+'/test_phot/{}'.format(os.path.basename(dir)))
            if os.path.isdir(i+'/test_phot/OUTPUT'):  shutil.rmtree(i+'/test_phot/OUTPUT')
            os.makedirs(i+'/test_phot/OUTPUT')

        if os.path.isfile(i+'/test_phot/cosmos.cat'):
            os.chdir(i+'/test_phot/')

            cmd = ['/home/epfl/sok/.local/fastpp/bin/fast++', 'fast.param']
            f = open('fast.log', 'w')
            subprocess.call(cmd)

            os.chdir(wdir)
        else:
            pass
