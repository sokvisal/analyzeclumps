import sys
sys.path.insert(0, './analyzeclumps')

%load_ext autoreload
%autoreload 1
%aimport segment
%aimport offset

from tqdm import tqdm
import glob
import os

tiles = ['t062']


wdir = "test_run"
os.mkdir(wdir)
os.copyfile('./analyzeclumps/segment.py', wdir)
os.copyfile('./analyzeclumps/offset.py', wdir)

os.chdir(wdir)

import segment
import offset

catdir = '/hpcstorage/sok/run/'
catname = 'cosmos_sfgs.dat'
decpath = '../cosmos/deconv'

for tile in tiles:
    offset.return_offsets(catdir, catname, tile, decpath)
    segment.wscat(catdir, 'cosmos_sfgs', '../deconv', tile, savedir=True)
