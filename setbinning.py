import misc
import glob
import diagnostics
from tqdm import tqdm
import numpy as np
import copy

def _return_photometry(directory, z):

    filtsets = ['subaru-rp', 'subaru-zp', 'ultravista-Y', 'ultravista-J', 'ultravista-H']
    if 0.5<z<=1.15:
        urest = filtsets[0]
    elif 1.15<z<=1.65:
        urest = filtsets[1]
    elif 1.65<z<=2.15:
        urest = filtsets[2]

    if 0.5<z<=0.75:
        vrest = filtsets[1]
    elif 0.75<z<=1.:
        vrest = filtsets[2]
    elif 1.<z<=1.6:
        vrest = filtsets[3]
    elif 1.6<z<=2.1:
        vrest = filtsets[4]

    y, x, signal, noise = np.loadtxt(directory+'/{}/vorbin_input.txt'.format(urest)).T #_lambd-000100.0
    # y, x, svrest, noise = np.loadtxt(directory+'/{}_lambd-000100.0/vorbin_input.txt'.format(vrest)).T
    # surest = surest.clip(0.01)
    # svrest = svrest.clip(0.01)
    return y, x, signal, noise #, svrest

def getnoise(directories, path, imgfact=False):
    import scipy.stats.mstats, scipy.optimize
    import random
    import matplotlib.pyplot as plt
    import os

    def aperture_rms(_dirs, iter, ):
        noise = []
        coord = []

        for fdir in _dirs:

            dataset = os.path.basename(fdir).split('-')[0]
            scalings = [10**(-6.4/2.5), 10**(-5/2.5)]
            if 'subaru' in dataset:
                scaling = scalings[0]
            else:
                scaling = scalings[1]

            gal, galh = diagnostics.openFits(fdir+'/g_1.fits')

            sexseg = misc._get_segmap(gal, fdir)[10:-10, 10:-10]
            segmap = np.zeros(sexseg.shape)
            segmap[sexseg==1] = np.nan

            dec, dech = diagnostics.openFits(fdir+'/deconv_01.fits')
            if imgfact:
                fact = diagnostics.search_cfg(fdir+'/config.py', 'IMG_FACT')
                dec *= float(fact[1:-1])
            segdec = dec.copy()[10:-10, 10:-10]

            # dec *= scaling
            y, x = np.indices(segdec.shape)
            ny, nx = y.ravel()[~np.isnan(segmap.ravel())], x.ravel()[~np.isnan(segmap.ravel())]

            h, w = dec.shape
            for i in np.arange(int(iter/len(_dirs))):
                ranint = random.randint(0,len(nx)-1)

                apermask = misc.createCircularMask(h, w, center=[ny[ranint]+10,nx[ranint]+10], radius=1.)
                masked_img = dec.copy()
                masked_img[~apermask] = 0

                # apermask = misc.createSquareMask(h, w, center=[ny[ranint]+10,nx[ranint]+10], width=3)
                # masked_img = dec.copy()
                # masked_img *= apermask

                # if '118778' in d: print np.count_nonzero(apermask)
                coord.append(ny[ranint]+10)
                coord.append(nx[ranint]+10)

                noise.append(np.nansum(masked_img))

        return noise

    tmpdirs = [fdir for _dirs in glob.glob('./{}/'.format(path)+directories) for fdir in sorted(glob.glob(_dirs+'/*-*'))][:]
    tmpdirs = np.array(tmpdirs).reshape(len(tmpdirs)/14, 14).T

    filternames = []
    sig = []

    for _dirs in tmpdirs:
        noise = aperture_rms(_dirs[:200], iter=2000., )

        r = [0.05,0.99]
        q = scipy.stats.mstats.mquantiles(noise, prob=[0, r[0], r[1], 1])
        h = np.histogram(noise, bins = np.linspace(q[1],q[2],100), normed=False)

        g = lambda c, sigma, I: lambda x: I*np.exp(-(x-c)**2/(2*sigma**2))
        errfun = lambda p: g(*p)(h[1][:h[0].shape[0]]) - h[0]
        bnind = np.where(h[0]==h[0].max())[0][0]
        p, success = scipy.optimize.leastsq(errfun,(h[1][bnind], np.abs(h[1][-1]-h[1][0])/20., h[0][bnind]), )
        p[1] = np.abs(p[1])
        sig.append(p[1])
        filternames.append(os.path.basename(_dirs[0]).split('-')[1].split('_')[0])

        # plt.figure(1)
        # plt.plot(h[1][:h[0].shape[0]], h[0], label='data distribution')
        # plt.plot(h[1][:h[0].shape[0]], errfun(p)+h[0], label='gaussian fit')
        # plt.plot(h[1][:h[0].shape[0]], errfun(p), label='error')
        # plt.show()

    # print (np.array(sig))
    # print (filternames)

    tile = directories.split('/')[1]
    np.savetxt('./{}/{}/noise.txt'.format(path, tile), sig, header = ' '.join([str(elem) for elem in filternames]) )


def getnoisedis(directories, path, deconvOffset=False, offset=False):
    import scipy.stats.mstats, scipy.optimize
    import pylab
    import glob
    import numpy as np

    from astropy.io import fits
    from astropy.io import ascii
    import matplotlib.pyplot as plt
    import warnings
    import os

    tile = directories.split('/')[1]
    sig = np.loadtxt('./{}/{}/noise.txt'.format(path, tile))
    print (sig)
    # print (1/0.)

    def _return_offseted_data(offsetsdata, filtname, data):

        bands = ['subaru_IA427', 'subaru_B', 'subaru_IA484', 'subaru_IA505', 'subaru_IA527', 'subaru_V',\
         'subaru_IA624', 'subaru_rp', 'subaru_IA738', 'subaru_zp', 'ultravista_Y', 'ultravista_J', 'ultravista_H']

        if filtname == 'ultravista_Ks':
            dy, dx = [0,0]
        else:
            idx_offset = bands[::-1].index(filtname)
            dy = offsetsdata['dy'].data[idx_offset]
            dx = offsetsdata['dx'].data[idx_offset]

        # print filtname, dy, dx
        data = np.roll(data, int(dy*3), 0)
        data = np.roll(data, int(dx*3), 1)
        # plt.imshow(data, origin='lower')
        # plt.show()
        return data

    warnings.simplefilter("error", RuntimeWarning)
    for d in glob.glob('./{}/'.format(path)+directories)[:]:
        _id = os.path.basename(d).split('_')[1].split('-')[1]
        # tile = d.split('/')[4][1:]

        segmap = fits.open('./{}/{}/watershed_segmaps/_id-{}.fits'.format(path, tile, _id))[0].data
        if deconvOffset: dec_offsets = ascii.read( './{}/offsets/_id-{}-dec.dat'.format(tile, int(_id)) )
        if offset: offsets = ascii.read( './{}/{}/offsets/_id-{}.dat'.format(path, tile, int(_id)) )


        for i, fdir in enumerate(sorted(glob.glob(d+'/*-*'))): #*.0
            filtname = os.path.basename(fdir).split('_')[0].replace('-', '_')
            dataset = os.path.basename(fdir).split('-')[0]

            scalings = [10**(-6.4/2.5), 10**(-5/2.5)]
            if 'subaru' in dataset:
                scaling = scalings[0]
                gain = 3.
            else:
                gain = 4.
                scaling = scalings[1]

            data = fits.open(fdir+'/deconv_01.fits')[0].data#*scaling
            data[segmap==0] = 0
            if deconvOffset: data = _return_offseted_data(dec_offsets, filtname, data)
            if offset: data = _return_offseted_data(offsets, filtname, data)
            zeroidx = np.where(data.ravel()!=0)[0]
            # print np.min(data), np.max(data)

            count = 0
            try:
                noise = np.ones(data.shape)*np.sqrt(data/gain+sig[i]**2)
            except RuntimeWarning :
                noise = np.ones(data.shape)*np.sqrt(sig[i]**2)
                count += 1
                if 'B' not in filtname: print ('aperture noise less than poison noise: '+filtname)

            y, x = np.indices(noise.shape)
            savedata = np.array([y.ravel()[zeroidx], x.ravel()[zeroidx], data.ravel()[zeroidx]*scaling, noise.ravel()[zeroidx]*scaling]).T

            # with open(fdir+'/vorbin_input.txt', 'w+') as datafile_id:
            #     np.savetxt(datafile_id, savedata, fmt=['%f','%f', '%f','%f'])

def create_cat(directories, path, constrain=False, bin_data=True):
    from vorbin.voronoi_2d_binning import voronoi_2d_binning
    from astropy.table import Table
    from astropy.io import ascii
    import numpy as np
    import shutil
    import os


    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    from scipy.signal import medfilt

    def _get_ellipse_params(segmapdir):
        from astropy.io import fits
        hdr = fits.open(segmapdir)[0].header

        return hdr['SEMIMAJ'], hdr['SEMIMIN'], hdr['PA']

    def _sn_func(index, signal=None, noise=None):
        # print index, signal, noise
        sn = np.sqrt(np.sum( (signal[index]/noise[index])**2. ))
        return sn

    def _save_cat(_dir, savefn, phot_param):
        if phot_param is not None:
            photcat_dir = _dir+'/test_phot'
            if os.path.exists(photcat_dir):
                shutil.rmtree(photcat_dir)
            os.makedirs(photcat_dir)

            z = float(os.path.basename(_dir).split('-')[-1])

            data = []
            for i in np.arange(phot_param.shape[0]):
                a = phot_param[i].ravel()
                a = np.insert(a,0,int(i))
                a = np.insert(a,len(a),z)
                data.append(a)
            data = np.array(data)
            # print (_dir, data.shape)
            photcat = Table([data[:,i] for i in range(30)], names=('id', 'F78', 'E78', 'F181', 'E181', 'F184', 'E184', 'F185', 'E185', 'F186', 'E186',\
                                                                                'F190', 'E190', 'F194', 'E194', 'F79', 'E79', 'F81', 'E81', 'F83', 'E83', 'F258', 'E258',\
                                                                                'F257', 'E257', 'F259', 'E259', 'F256', 'E256', 'z_spec'),\
                                                                                meta={'name': 'cosmos sfg catalog'})
            for colname in photcat.colnames[1:]:
                photcat[colname].format = '%6.5e'
            ascii.write(photcat, photcat_dir+savefn, overwrite=True,  format='commented_header')

    def pix2pix(directory):
        d = directory
        if os.path.isfile(d+'/vorbin_output.txt'):
            newdir = [i+'/vorbin_input.txt' for i in sorted(glob.glob(d+'/*.0'))]

            for i, d in enumerate(newdir):
                y,x,signal,noise = np.loadtxt(d).T
                signal = signal.clip(0.)

                if i == 0:
                    phot_param = zip(signal, noise)
                else:
                    phot_param = np.concatenate((phot_param, zip(signal,noise)), axis=1)

            return phot_param
        else:
            return None

    def unravel_map(d, y,x, raveled_signal, raveled_bsig, sn, secsn, size=156):
        from skimage.measure import label
        tile = d.split('/')[4][1:]
        _id = os.path.basename(d).split('-')[1].split('_')[0]

        def _ellipse(masked, center, a, b, phi):
            yi, xi = np.indices(masked.shape)
            yi = yi.ravel()[~np.isnan(masked).ravel()]
            xi = xi.ravel()[~np.isnan(masked).ravel()]

            xc = center[0]
            yc = center[1]

            ell = ((xi-xc)*np.cos(phi)+(yi-yc)*np.sin(phi))**2./a**2 + ((xi-xc)*np.sin(phi)-(yi-yc)*np.cos(phi))**2./b**2

            tmpidx = np.where(ell<1)[0]
            return len(tmpidx)

        minsn = 0.5
        unraveled_maps = np.zeros((2,size,size))*np.nan
        unique_map = np.zeros((size,size))
        coords = zip(y.astype(int),x.astype(int))
        tmpdata = [raveled_signal[sn>minsn], raveled_bsig[secsn>minsn]]
        tmpsn = [sn, secsn]

        # a, b, phi = _get_ellipse_params('../run/images/{}/watershed_segmaps/_id-{}.fits'.format(tile, _id))
        a, b, phi = _get_ellipse_params('./{}/{}/watershed_segmaps/_id-{}.fits'.format(path, tile, _id))

        for i, raveled_data in enumerate(tmpdata):
            for j, (yi,xi) in enumerate(zip(y[tmpsn[i]>minsn],x[tmpsn[i]>minsn])):
                unraveled_maps[i, int(yi),int(xi)] = raveled_data[j]
                unique_map[int(yi),int(xi)] = 1.

        labels = label(unique_map, neighbors=4)
        seg = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        seg = seg.astype(float)
        seg = medfilt(seg)

        seg[seg>0] = 1.
        seg = ndi.binary_fill_holes(seg).astype(float)
        seg[seg==0] = np.nan

        # oldcounts = 0
        # newa = a/4.
        # e = b/a
        # while True:
        #     counts = _ellipse(seg, [77,77], newa, e*newa, phi)
        #     if counts>oldcounts:
        #         newa += 1.
        #         oldcounts = counts
        #     else:
        #         break

        plt.subplot(1,1,1)
        plt.imshow(seg, origin='lower')
        plt.show()
        newy, newx = np.indices(seg.shape)
        newy = newy.ravel()[~np.isnan(seg.ravel())]
        newx = newx.ravel()[~np.isnan(seg.ravel())]
        indices = [i for i, coord in enumerate(zip(y,x)) if coord in zip(newy, newx)]
        outidx = [i for i, coord in enumerate(zip(y,x)) if coord not in zip(newy, newx)]

        # plt.close()
        return indices, outidx

    def vbin(directory):
        d = directory

        z = float(os.path.basename(d).split('-')[2])
        # y, x, bsig, bn = _return_photometry(d, z)
        y, x, bsig, bn = np.loadtxt(d+'/subaru-rp/vorbin_input.txt').T
        y, x, signal, noise = np.loadtxt(d+'/ultravista-H/vorbin_input.txt').T #_lambd-000100.0 #ultravista-H
        targetSN = 5.
        sn = (signal)/np.sqrt(noise**2)
        secsn = bsig/np.sqrt(bn**2)
        # sn += secsn

        # indices, outidx = unravel_map(d, y,x,signal,bsig, sn, secsn)
        # vars = [x,y,signal,noise,bsig,bn]
        # innerx, innery, innersig, innernoise, innerbsig, innerbn = [var[indices] for var in vars]
        # outerx, outery, outersig, outernoise, outerbsig, outerbn = [var[outidx] for var in vars]

        if not constrain:
            try:
                targetSN = 5.
                outbinNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x, y, signal, noise, targetSN,\
                                                                        pixelsize=1., plot=1, quiet=1, cvt=1, wvt=1, sn_func=_sn_func) #bphot=[bsig, bn],
            except:
                targetSN = 3.
                outbinNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x, y, signal, noise, targetSN,\
                                                                        pixelsize=1., plot=1, quiet=1, cvt=1, wvt=1, sn_func=_sn_func, secsignal=None, secnoise=None)
        else:
            try:
                outbinNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x, y, signal, noise, targetSN,\
                                                                        pixelsize=1., plot=1, quiet=1, cvt=1, wvt=1, sn_func=_sn_func, secsignal=bsig, secnoise=bn) #bphot=[bsig, bn],
            except:
                targetSN = 3.
                outbinNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x, y, signal, noise, targetSN,\
                                                                        pixelsize=1., plot=1, quiet=1, cvt=1, wvt=1, sn_func=_sn_func, secsignal=bsig, secnoise=bn) #bphot=[bsig, bn],
                print ('no signal, going to s/n of 3, ', max(outbinNum), d)

        print (max(outbinNum), d)
        binNum = outbinNum

        # if len(nPixels[nPixels==1]) > 15:
        #     print len(nPixels[nPixels==1])
        #     if 15<len(nPixels[nPixels==1])<30:
        #         targetSN = 10.
        #     else:
        #         targetSN = 15.
        #     idx = []
        #     orderedidx = []
        #     for ind in range(min(outbinNum), max(outbinNum)+1):
        #         if len(outbinNum[outbinNum==ind]) == 1:
        #             idx.append(list(outbinNum).index(ind))
        #         else:
        #             orderedidx.append(np.where(outbinNum==ind)[0])
        #     cenbinNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(x[idx], y[idx], signal[idx], noise[idx], targetSN,\
        #                                                     pixelsize=1., plot=0, quiet=1, cvt=1, wvt=1, sn_func=_sn_func) #bphot=[bsig, bn],
        #
        #     for i, idxbin in enumerate(orderedidx):
        #         outbinNum[idxbin] = i+max(cenbinNum)
        #
        #     binNum = outbinNum
        #     binNum[idx] = cenbinNum
        #     print 'reduced to: '+ str(max(binNum))
        # else:
        #     print 'number of unbinned pixels: '+str(len(nPixels[nPixels==1]))
        #     binNum = outbinNum

        # _id = os.path.basename(d).split('-')[1].split('_')[0]
        # np.savetxt('./2channels-binning/id{}-vout{}-.txt'.format(_id, int(constrain)), np.c_[binNum])

        _id = os.path.basename(d).split('-')[1].split('_')[0]
        np.savetxt(d+'/vorbin_output.txt', np.c_[binNum])
        if os.path.isfile(d+'/vorbin_output.txt'):
            newdir = [i+'/vorbin_input.txt' for i in sorted(glob.glob(d+'/*-*'))] #*.0

            signals, noises = [], []
            for tmpdir in newdir:
                y, x, s, n= np.loadtxt(tmpdir).T
                signals.append(s)
                noises.append(n)

            phot_param = []
            for binid in np.arange(np.max(binNum)+1):
                idx = np.where(binNum==binid)[0]

                ffe = [] # flux and flux error
                for j in np.arange(len(newdir)):
                    meanphot = np.mean(signals[j][idx])
                    if meanphot < 0: meanphot=0.0
                    ffe.append([ meanphot, np.sqrt(np.sum(noises[j][idx]**2))/len(signals[j][idx]) ]) #max(np.sqrt(np.sum(noises[j][idx]**2)), np.std(signals[j][idx]))])

                phot_param.append(ffe)
            phot_param = np.array(phot_param)
            return phot_param
        else:
            return None

    nosnr = []
    # print list(glob.glob('./{}/'.format(path)+directories)[:]).index('././deconv/./at065/_id-119634_z-0.817')
    # print 1/0.
    _dirs = [idnames for idnames in glob.glob('./{}/'.format(path)+directories) if len(glob.glob(idnames+'/*-*'))==14]
    # print _dirs
    for d in tqdm(_dirs[:]):
        phot_param = vbin(d) # if voronoi binning
        _save_cat(d, '/cosmos.cat', phot_param)

        # phot_param = pix2pix(d)
        # _save_cat(d, '/cosmos_p2p.cat', phot_param)
