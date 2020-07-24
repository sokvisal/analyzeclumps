from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii
import astropy.wcs as pywcs
from astropy.stats import sigma_clip
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

import sys
sys.path.insert(0, '/mnt/drivea/test/')

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.filters import roberts
from scipy.signal import correlate
from skimage.transform import rescale
from skimage.feature import peak_local_max


from numpy.linalg import eig, inv

def wscat(catdir, catname, path, tile, savedir=False):
    # load full catalog
    allobjs = np.loadtxt(catdir+'/cosmos_full_cat.dat', skiprows=1) #cosmos_full_cat
    objra = allobjs[:,2]
    objdec = allobjs[:,3]

    # load sfg catalogs
    table = ascii.read('{}.dat'.format(catdir+catname)) #cosmos_sfgcat_extended_update
    try:
        catalogs = np.array([table['z'], table['id'], table['x'], table['y'], table['ra'], table['dec']]).T
    except KeyError:
        catalogs = np.array([table['zphot'], table['id'], table['x'], table['y'], table['ra'], table['dec']]).T
    _ids = list(catalogs[:,1])#np.loadtxt('{}.dat'.format(catname), skiprows=1, usecols=[0])
    # _ids = list(_ids)

    idsdec = [os.path.basename(_dir).split('-')[1].split('_')[0] for _dir in glob.glob('./{}/a{}/_id-*'.format(path, tile))]
    idx = [_ids.index(float(_iddec)) for _iddec in idsdec]
    catalogs = catalogs[idx]
    catra = catalogs[:,4]
    catdec = catalogs[:,5]
    _ids = list(catalogs[:,1])

    # load tile image
    hdu = fits.open(catdir+'/images/{}/{}-ultravista_Ks.fits'.format(tile, tile))
    imgks = hdu[0].data
    header = hdu[0].header
    hdu.close()
    wcs = pywcs.WCS(header)

    # check extend of tile image
    tmparray = np.arange(31,4096-31)
    ra, dec = wcs.all_pix2world(tmparray, tmparray, 1, ra_dec_order=True)
    minra = min(ra)
    maxra = max(ra)
    mindec = min(dec)
    maxdec = max(dec)

    # return idx of sfgs within tile image
    ra_idx = np.where((catra>minra)&(catra<maxra))[0]
    dec_idx = np.where((catdec>mindec)&(catdec<maxdec))[0]
    dec_idx = set(dec_idx)
    idx =  list(set(ra_idx).intersection(set(dec_idx)))

    # change ra and dec to pixel coordinate of tile image
    catx, caty = wcs.all_world2pix(catra[idx], catdec[idx], 1, ra_dec_order=True)
    catalogs = catalogs[idx, :]
    catalogs[:,2] = catx
    catalogs[:,3] = caty

    catalogs = catalogs[:]
    print (catalogs.shape)
    for (_id,x,y) in zip(catalogs[:,1], catalogs[:,2], catalogs[:,3]):

        size = 26
        scale = 3
        tmpobjs = np.array(return_objcoord(wcs, [int(y), int(x)], size, [objra, objdec], scale=scale))
        localpeaks = peaks(tmpobjs, (2*size*scale,2*size*scale))
        markers = ndi.label(localpeaks)[0]

        matchimg = imgks[int(y)-size:int(y)+size, int(x)-size:int(x)+size]
        matchimg = rescale(matchimg, 156./52., mode='reflect', multichannel=False)*(52./156.)**2
    #     tmpdir = glob.glob('./selectedgal/resolved_sfgs/a{}/_id-{}*'.format(tile, int(_id)))[0]
    #     matchimg = fits.open('{}/subaru-zp_lambd-000100.0/g_1.fits'.format(tmpdir))[0].data
    #     matchimg = rescale(matchimg, 156./52., mode='reflect', multichannel=False)*(52./156.)**2

        tmpsig = 3.5
        while True:
            masked = matchimg.copy()
            filtered = sigma_clip(matchimg, sigma=tmpsig, masked=True)
            distance = filtered.data
            filtered = filtered.mask
            masked[filtered==False] = np.nan
            if not np.isnan(masked[77,77]):
                labels = watershed(-distance/np.sum(distance), markers, compactness=0.1, mask=filtered) #-masked/np.sum(masked)

                segmap = labels.copy()
                segmap[abs(segmap-segmap[77,77])>0] = 0.
                segmap = ndi.binary_fill_holes(segmap).astype(float)

                edges = roberts(segmap)
                edges[edges>0] = 1

                ex = np.where(edges==1)[1]
                ey = np.where(edges==1)[0]
                fit = fitEllipse(ex, ey, segmap)
                tmpx, tmpy, a, b, phi = fit.getparams()
                if not np.any(np.isnan([a,b,phi])):
                    break
            tmpsig -= 0.5

        ell = fit.plotEllipse(a, b, phi)

        # print (tmpsig,a,b,phi)
        # nrow = 1
        # ncol = 4
        # imscale = 2.5
        # fig = plt.figure(figsize=((ncol+1)*imscale, (nrow+1)*imscale))
        # gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1),\
        #                    bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1))
        #
        # ax = plt.subplot(gs[0])
        # ax.imshow(matchimg, origin='lower')
        #
        # ax = plt.subplot(gs[1])
        # ax.imshow(masked, origin='lower')
        # ax.scatter(tmpobjs[:,1],tmpobjs[:,0], marker='x', color='r')
        #
        # ax = plt.subplot(gs[2])
        # ax.imshow(segmap, cmap=plt.cm.nipy_spectral, origin='lower')
        # ax.scatter(tmpobjs[:,1],tmpobjs[:,0], marker='x', color='r')
        #
        # ax = plt.subplot(gs[3])
        # ax.imshow(edges, origin='lower')
        # ax.scatter(tmpx+ell[0,:], tmpy+ell[1,:], color='g', s=2)
        # plt.show()

        if not os.path.isdir('./{}/a{}/watershed_segmaps'.format(path, tile)):
            os.makedirs('./{}/a{}/watershed_segmaps'.format(path, tile))

        if savedir:
            hdr = fits.Header()
            hdr['XC'] = tmpx
            hdr['YC'] = tmpy
            hdr['SEMIMAJ'] = a
            hdr['SEMIMIN'] = b
            hdr['AXISUNIT'] = 'arcsecond'
            hdr['PA'] = phi
            hdr['PAUNIT'] = 'rad'
            hdr['SEGMAP'] = 'Based on Ks'
            hdu = fits.PrimaryHDU(segmap, hdr)
            hdu.writeto('./{}/a{}/watershed_segmaps/_id-{}.fits'.format(path, tile, int(_id)), overwrite=True)

def _get_photometric(z):

    filtsets = ['subaru_rp', 'subaru_zp', 'ultravista_Y', 'ultravista_J', 'ultravista_H']
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

    return urest #, svrest

def ws_2maps(catname, path, tile, savedir=False):
    # load full catalog
    allobjs = np.loadtxt('/mnt/drivea/run/cosmos_catalogs/cosmos_full_cat.dat', skiprows=1) #cosmos_full_cat
    objra = allobjs[:,2]
    objdec = allobjs[:,3]

    # load sfg catalogs
    table = ascii.read('{}.dat'.format(catname)) #cosmos_sfgcat_extended_update
    try:
        catalogs = np.array([table['z'], table['id'], table['x'], table['y'], table['ra'], table['dec']]).T
    except KeyError:
        catalogs = np.array([table['zphot'], table['id'], table['x'], table['y'], table['ra'], table['dec']]).T
    _ids = list(catalogs[:,1])#np.loadtxt('{}.dat'.format(catname), skiprows=1, usecols=[0])
    # _ids = list(_ids)

    idsdec = [os.path.basename(_dir).split('-')[1].split('_')[0] for _dir in glob.glob('./{}/a{}/_id-*'.format(path, tile))]
    idx = [_ids.index(float(_iddec)) for _iddec in idsdec]
    catalogs = catalogs[idx]
    catra = catalogs[:,4]
    catdec = catalogs[:,5]
    _ids = list(catalogs[:,1])

    # load tile image
    hdu = fits.open('/mnt/drivea/run/images/{}/{}-ultravista_Ks.fits'.format(tile, tile))
    imgks = hdu[0].data
    header = hdu[0].header
    wcs = pywcs.WCS(header)

    # check extend of tile image
    tmparray = np.arange(31,4096-31)
    ra, dec = wcs.all_pix2world(tmparray, tmparray, 1, ra_dec_order=True)
    minra = min(ra)
    maxra = max(ra)
    mindec = min(dec)
    maxdec = max(dec)

    # return idx of sfgs within tile image
    ra_idx = np.where((catra>minra)&(catra<maxra))[0]
    dec_idx = np.where((catdec>mindec)&(catdec<maxdec))[0]
    dec_idx = set(dec_idx)
    idx =  list(set(ra_idx).intersection(set(dec_idx)))

    # change ra and dec to pixel coordinate of tile image
    catx, caty = wcs.all_world2pix(catra[idx], catdec[idx], 1, ra_dec_order=True)
    catalogs = catalogs[idx, :]
    catalogs[:,2] = catx
    catalogs[:,3] = caty

    catalogs = catalogs[:]
    print (catalogs.shape)
    for (_id,x,y,zphot) in zip(catalogs[:,1], catalogs[:,2], catalogs[:,3], catalogs[:,0]):

        size = 26
        scale = 3
        tmpobjs = np.array(return_objcoord(wcs, [int(y), int(x)], size, [objra, objdec], scale=scale))
        localpeaks = peaks(tmpobjs, (2*size*scale,2*size*scale))
        markers = ndi.label(localpeaks)[0]

        matchimg = imgks[int(y)-size:int(y)+size, int(x)-size:int(x)+size]
        matchimg = rescale(matchimg, 156./52., mode='reflect', multichannel=False)*(52./156.)**2


        photo_fn = _get_photometric(zphot)
        imgb = fits.open('/mnt/drivea/run/images/{}/{}-subaru_rp.fits'.format(tile, tile))[0].data
        matchb = imgb[int(y)-size:int(y)+size, int(x)-size:int(x)+size]
        matchb = rescale(matchb, 156./52., mode='reflect', multichannel=False)*(52./156.)**2
    #     tmpdir = glob.glob('./selectedgal/resolved_sfgs/a{}/_id-{}*'.format(tile, int(_id)))[0]
    #     matchimg = fits.open('{}/subaru-zp_lambd-000100.0/g_1.fits'.format(tmpdir))[0].data
    #     matchimg = rescale(matchimg, 156./52., mode='reflect', multichannel=False)*(52./156.)**2

        tmpsig = 3.
        while True:
            masked = matchimg.copy()
            filtered = sigma_clip(matchimg, sigma=tmpsig, masked=True)
            distance = filtered.data
            filtered = filtered.mask
            masked[filtered==False] = np.nan

            mb = matchb.copy()
            fb = sigma_clip(matchb, sigma=tmpsig, masked=True)
            distb = fb.data
            fb = fb.mask
            mb[fb==False] = np.nan

            if not np.isnan(masked[77,77]):
                labels = watershed(-distance/np.sum(distance), markers, compactness=0.1, mask=filtered) #-masked/np.sum(masked)
                sm = labels.copy()
                sm[abs(sm-sm[77,77])>0] = 0.
                sm = ndi.binary_fill_holes(sm).astype(float)

                labels = watershed(-distb/np.sum(distb), markers, compactness=0.1, mask=fb) #-masked/np.sum(masked)
                smb = labels.copy()
                smb[abs(smb-smb[77,77])>0] = 0.
                smb = ndi.binary_fill_holes(smb).astype(float)

                segmap = smb+sm
                segmap[segmap>0] = 1

                # plt.imshow(sm)
                # plt.show()
                #
                # plt.imshow(smb-sm)
                # plt.show()

                edges = roberts(segmap)
                edges[edges>0] = 1

                ex = np.where(edges==1)[1]
                ey = np.where(edges==1)[0]
                fit = fitEllipse(ex, ey, sm)
                tmpx, tmpy, a, b, phi = fit.getparams()
                if not np.any(np.isnan([a,b,phi])):
                    break
            tmpsig -= 0.5

        ell = fit.plotEllipse(a, b, phi)

        # if _id == 114072:
        # print (_id,tmpsig,a,b,phi)
        # nrow = 1
        # ncol = 4
        # imscale = 2.5
        # fig = plt.figure(figsize=((ncol+1)*imscale, (nrow+1)*imscale))
        # gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1),\
        #                    bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1))
        #
        # ax = plt.subplot(gs[0])
        # ax.imshow(matchimg, origin='lower')
        #
        # ax = plt.subplot(gs[1])
        # ax.imshow(masked, origin='lower')
        # ax.scatter(tmpobjs[:,1],tmpobjs[:,0], marker='x', color='r')
        #
        # ax = plt.subplot(gs[2])
        # ax.imshow(segmap, cmap=plt.cm.nipy_spectral, origin='lower')
        # ax.scatter(tmpobjs[:,1],tmpobjs[:,0], marker='x', color='r')
        #
        # ax = plt.subplot(gs[3])
        # ax.imshow(edges, origin='lower')
        # ax.scatter(tmpx+ell[0,:], tmpy+ell[1,:], color='g', s=2)
        # plt.show()

        if savedir:
            if not os.path.isdir('./{}/a{}/watershed_segmaps'.format(path, tile)):
                os.makedirs('./{}/a{}/watershed_segmaps'.format(path, tile))

            hdr = fits.Header()
            hdr['XC'] = tmpx
            hdr['YC'] = tmpy
            hdr['SEMIMAJ'] = a
            hdr['SEMIMIN'] = b
            hdr['AXISUNIT'] = 'arcsecond'
            hdr['PA'] = phi
            hdr['PAUNIT'] = 'rad'
            hdr['SEGMAP'] = 'Based on Ks and r+'
            hdu = fits.PrimaryHDU(segmap, hdr)
            hdu.writeto('./{}/a{}/watershed_segmaps/_id-{}.fits'.format(path, tile, int(_id)), overwrite=True)

class fitEllipse:
    def __init__(self, x, y, segmap):
        self.segmap = segmap

        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:,n]

        self.a = a

    def ellipse_center(self):
        a = self.a

        b,c,d,f,g,a = a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        return [x0,y0]

    def ellipse_axis_length(self):
        a = self.a

        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)

        return np.array([res1, res2])

    def ellipse_angle_of_rotation(self):
        a = self.a

        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        if b == 0:
            if a < c:
                return 0
            else:
                return np.pi/2
        else:
            if a < c:
                return np.arctan(2*b/(a-c))/2
            else:
                return np.pi/2 + np.arctan(2*b/(a-c))/2

    def getparams(self):
        def _ellipse(masked, center, a, b, phi):
            yi, xi = np.indices(masked.shape)
            yi = yi.ravel()[~np.isnan(masked).ravel()]
            xi = xi.ravel()[~np.isnan(masked).ravel()]

            xc = center[0]
            yc = center[1]

            ell = ((xi-xc)*np.cos(phi)+(yi-yc)*np.sin(phi))**2./a**2 + ((xi-xc)*np.sin(phi)-(yi-yc)*np.cos(phi))**2./b**2

            tmpidx = np.where(ell<1)[0]
            return len(tmpidx)

        center = self.ellipse_center()
        phi = self.ellipse_angle_of_rotation()
        axes = self.ellipse_axis_length()

        ## Check which orientation contains more pixels from the segmap and return the parameters for the
        ## correct orientaiton
        masked = np.zeros(self.segmap.shape)
        masked[self.segmap==0] = np.nan


        a = np.max(axes)
        b = np.min(axes)

        if _ellipse(masked, center, a,b, phi) > _ellipse(masked, center, a,b, phi-np.deg2rad(90)):
            pass
        else:
            phi = phi-np.deg2rad(90)

        self.axmaj = a
        self.axmin = b
        self.phi = phi
        return center+[self.axmaj, self.axmin, phi]

    def plotEllipse(self, a, b, to):
        a = self.axmaj
        b = self.axmin
        phi = self.phi

        t = np.linspace(0, 2*np.pi, 100)
        Ell = np.array([a*np.cos(t) , b*np.sin(t)])
        R_rot = np.array([[np.cos(to) , -np.sin(to)],[np.sin(to) , np.cos(to)]])

        Ell_rot = np.zeros((2,Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

        return Ell_rot

def getSNR(_id, idx):
    fluxes = [1,2,3,4,13,6,12,7,11,15,10,8,14] # making sure to retrieve the right flux
    id_idx = _ids.index(_id)
    return snrcat[id_idx, fluxes[idx]]

def return_objcoord(wcs, center, size, objcoords, scale=3):
    y, x = center
    dy = 4096/2-y
    dx = 4096/2-x

    import astropy.wcs as pywcs
    yarray = np.arange(y-size,y+size)
    xarray = np.arange(x-size,x+size)
    ra, dec = wcs.all_pix2world(xarray, yarray, 1, ra_dec_order=True)
    minra = min(ra)
    maxra = max(ra)
    mindec = min(dec)
    maxdec = max(dec)

    pixelcoords = []
    for tmpra, tmpdec in zip(objcoords[0], objcoords[1]):
        if (minra<tmpra<maxra) and (mindec<tmpdec<maxdec):
            xc, yc = wcs.all_world2pix(tmpra, tmpdec, 1, ra_dec_order=True)
            dy = (yc-y+size)*scale
            dx = (xc-x+size)*scale
            pixelcoords.append([dy,dx])
    return np.array(pixelcoords)

def peaks(pixelcoords, shape):
    array = np.zeros(shape, dtype=bool)

    for y, x in zip(pixelcoords[:,0], pixelcoords[:,1]):
        array[int(y),int(x)] = True

    return array
