import os
import glob
from tqdm import tqdm
import subprocess
import shutil

def fit(path, tile):
    directories = [idnames for idnames in glob.glob('{}/a{}/_id-*'.format(path, tile)) ] #'../run/selectedgal/resolved_sfgs/at064/_id*' #'../run/at065/_id*'
    print ('Starting queue')
    wdir = os.getcwd()

    script_dir = '/hpcstorage/sok/run/cosmos/analyzeclumps'
    for i in tqdm(directories[:200]):

        if os.path.isdir(i+'/test_phot'):
            for dir in glob.glob('../analyzeclumps/sedfiles/*.param'):
                shutil.copyfile(dir, i+'/test_phot/{}'.format(os.path.basename(dir)))
            if os.path.islink(i+'/test_phot/templates'): os.unlink(i+'/test_phot/templates')
            os.system('ln -s {}/sedfiles/templates {}/test_phot/templates'.format(script_dir, i))
            if os.path.isdir(i+'/test_phot/OUTPUT'):  shutil.rmtree(i+'/test_phot/OUTPUT')
            os.makedirs(i+'/test_phot/OUTPUT')

        if os.path.isfile(i+'/test_phot/cosmos.cat'):
            os.chdir(i+'/test_phot/')

            # zphotparams = ['zphot_2800-u.param', 'zphot_u-v.param']
            # for param in zphotparams:
            #     shutil.copyfile(param, 'zphot.param')
            #     ezycmd = ['/home/astro/sok/.local/eazy-photoz/src/eazy', 'zphot.param']
            #
            #     f = open('eazy.log', 'w')
            #     subprocess.call(ezycmd, stdout=f)

            cmd = ['/home/astro/sok/.local/fastpp/bin/fast++', 'fast.param']
            f = open('fast.log', 'w')
            subprocess.call(cmd)

            os.chdir(wdir)
        else:
            pass
