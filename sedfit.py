import os
import glob
from tqdm import tqdm
import subprocess
import shutil
tilelist =  ['at086', 'at038', 'at039', 'at040', 'at041', 'at042', 'at043', 'at044',\
             'at027', 'at028', 'at029']

for tile in tilelist:

    directories = './selectedgal/resolved_sfgs_fastpp/{}/_id-*'.format(tile) #'../run/selectedgal/resolved_sfgs/at064/_id*' #'../run/at065/_id*'
    # print ('Starting queue')

    for i in tqdm(glob.glob(directories)[:]):

        if not os.path.isdir(i+'/test_phot/fast.param'):

            for dir in glob.glob('./extras/sedfiles/*.param'):
                shutil.copyfile(dir, i+'/test_phot/{}'.format(os.path.basename(dir)))
            # if os.path.isdir(i+'/test_phot/templates'): shutil.rmtree(i+'/test_phot/templates')
            # shutil.copytree('./sedfiles/templates', i+'/test_phot/templates')
            if os.path.isdir(i+'/test_phot/OUTPUT'):  shutil.rmtree(i+'/test_phot/OUTPUT')
            os.makedirs(i+'/test_phot/OUTPUT')
            if os.path.islink('{}/test_phot/templates'.format(i)): os.unlink('{}/test_phot/templates'.format(i))
            os.symlink('/mnt/drivea/run/extras/sedfiles/templates', '{}/test_phot/templates'.format(i))

        if os.path.isfile(i+'/test_phot/cosmos.cat'):
            os.chdir(i+'/test_phot/')

            zphotparams = ['zphot_2800-u.param', 'zphot_u-v.param']
            for param in zphotparams:
                shutil.copyfile(param, 'zphot.param')
                ezycmd = ['/mnt/drivea/run/eazy-photoz/src/eazy', 'zphot.param']

                f = open('eazy.log', 'w')
                subprocess.call(ezycmd, stdout=f)
            # print 'Copying zout...'
            shutil.move('./OUTPUT/cosmos.zout', './cosmos.zout')

            cmd = ['/usr/local/fastpp/bin/fast++', 'fast.param']
            # cmd = ['/mnt/drivea/run/FAST_v1.0/superfast_idl81', 'fast.param']
            f = open('fast.log', 'w')
            subprocess.call(cmd)

            os.chdir('/mnt/drivea/run/')
        #
        # # print i
        # if os.path.isfile('/mnt/drivea/run/'+i+'/test_phot/cosmos.cat'):
        #     os.chdir('/mnt/drivea/run/'+i+'/test_phot/')
        #
        #     zphotparams = ['zphot_2800-u.param', 'zphot_u-v.param']
        #     for param in zphotparams:
        #         shutil.copyfile(param, 'zphot.param')
        #         ezycmd = ['/mnt/drivea/run/eazy-photoz/src/eazy', 'zphot.param']
        #
        #         f = open('eazy.log', 'w')
        #         subprocess.call(ezycmd, stdout=f)
        #
        #     # print 'Copying zout...'
        #     shutil.move('/mnt/drivea/run/'+i+'/test_phot/OUTPUT/cosmos.zout', '/mnt/drivea/run/'+i+'/test_phot/cosmos.zout')
        #
        #     cmd = ['/mnt/drivea/run/FAST_v1.0/superfast_idl81', 'fast.param']
        #     # f = open('fast.log', 'w')
        #     subprocess.call(cmd)
        #     # os.system('../../../../../test/FAST_v1.0/fast fast.param > {}/output.log'.format(i+'/test_phot'))

        else:
            pass
