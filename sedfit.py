import os
import glob
from tqdm import tqdm
import subprocess
import shutil

tiles = [ '../cosmos/deconv/at065'] #['at066', 'at075', 'at076', 'at077']
# tiles = ['../cosmos/deconv/at062', '../cosmos/deconv/at063', '../cosmos/deconv/at064',  '../cosmos/deconv/at065', '../deconv/at066', '../deconv/at075', '../deconv/at076', '../deconv/at077']
wdir = os.getcwd()

for tile in tiles:
    directories = [idnames for idnames in glob.glob('{}/_id-*'.format(tile)) if len(glob.glob(idnames+'/*-*'))==14] #'../run/selectedgal/resolved_sfgs/at064/_id*' #'../run/at065/_id*'
    print ('Starting queue')

    for i in tqdm(directories[212:213]):

        if not os.path.isfile(i+'/test_phot/fast.param'):
            for dir in glob.glob('./sedfiles/*.param'):
                shutil.copyfile(dir, i+'/test_phot/{}'.format(os.path.basename(dir)))
            # if os.path.isdir(i+'/test_phot/templates'): shutil.rmtree(i+'/test_phot/templates')
            # shutil.copytree('./sedfiles/templates', i+'/test_phot/templates')
            if os.path.isdir(i+'/test_phot/OUTPUT'):  shutil.rmtree(i+'/test_phot/OUTPUT')
            os.makedirs(i+'/test_phot/OUTPUT')

        if os.path.isfile(i+'/test_phot/cosmos.cat'):
            os.chdir(i+'/test_phot/')

            # zphotparams = ['zphot_2800-u.param', 'zphot_u-v.param']
            # for param in zphotparams:
            #     shutil.copyfile(param, 'zphot.param')
            #     ezycmd = ['/mnt/drivea/run/eazy-photoz/src/eazy', 'zphot.param']
            #
            #     f = open('eazy.log', 'w')
            #     subprocess.call(ezycmd, stdout=f)
            #
            # # print 'Copying zout...'
            # shutil.move('./OUTPUT/cosmos.zout', './cosmos.zout')

            cmd = ['/usr/local/fastpp/bin/fast++', 'fast.param']
            # cmd = ['/mnt/drivea/run/FAST_v1.0/superfast_idl81', 'fast.param']
            f = open('fast.log', 'w')
            subprocess.call(cmd)

            os.chdir(wdir)
        else:
            pass

def fit(path, tile):
    directories = [idnames for idnames in glob.glob('{}/a{}/_id-*'.format(path, tile)) ] #'../run/selectedgal/resolved_sfgs/at064/_id*' #'../run/at065/_id*'
    print ('Starting queue')
    wdir = os.getcwd()

    for i in directories[212:213]:

        if not os.path.isfile(i+'/test_phot/fast.param'):
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
