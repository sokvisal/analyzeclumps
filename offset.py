bands = ['subaru_IA427', 'subaru_B', 'subaru_IA484', 'subaru_IA505', 'subaru_IA527', 'subaru_V',\
         'subaru_IA624', 'subaru_rp', 'subaru_IA738', 'subaru_zp', 'ultravista_Y', 'ultravista_J', 'ultravista_H']

from astropy.io import fits
from astropy.table import Table
import astropy.wcs as pywcs
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import glob
import os
import shutil
from tqdm import tqdm

import sys
sys.path.insert(0, '../../test/')

from scipy.signal import correlate

def create_rgb(b, g, r, scalemin, scale='linear'):
    import img_scale
    img = np.zeros((r.shape[0], r.shape[1], 3), dtype=float)

    img[:,:,2] = getattr(img_scale, scale)(b, scale_min=scalemin[0], scale_max=np.max(b)*1.2)
    img[:,:,1] = getattr(img_scale, scale)(g, scale_min=scalemin[1], scale_max=np.max(g))
    img[:,:,0] = getattr(img_scale, scale)(r, scale_min=scalemin[2], scale_max=np.max(r))

    return img

# catname = 'cosmos_sfgs.dat' #cosmos_close40
# path = './deconv'

def get_offsets(catdir, catname, tile, path):
    from skimage.transform import resize
    from skimage import img_as_bool
    from scipy.signal import correlate

    if not os.path.isdir('./{}/a{}/offsets'.format(path, tile)):
        os.makedirs('./{}/a{}/offsets'.format(path, tile))

    tmpcat = np.loadtxt(catdir+catname, skiprows=1, usecols=[i for i in np.arange(11,43)])
    _ids = np.loadtxt(catdir+catname, skiprows=1, usecols=[0])
    _ids = list(_ids)
    snrs = 10**((tmpcat[:, ::2]-tmpcat[:, 1::2])/-2.5)

    def getSNR(_id, idx, snrcat):
        fluxes = [1,2,3,4,13,6,12,7,11,15,10,8,14] # making sure to retrieve the right flux
        id_idx = _ids.index(_id)
        return snrcat[id_idx, fluxes[idx]]

    bands = ['subaru-IA427', 'subaru-B', 'subaru-IA484', 'subaru-IA505', 'subaru-IA527', 'subaru-V', 'subaru-IA624',\
             'subaru-rp', 'subaru-IA738', 'subaru-zp', 'ultravista-Y', 'ultravista-J', 'ultravista-H']

    for dirname in tqdm(glob.glob('./{}/a{}/_id*'.format(path, tile))[:]):
        # print (dirname)
        _id = int(os.path.basename(dirname).split('-')[1].split('_')[0])
        idx = _ids.index(_id)

        hdu = fits.open(glob.glob('{}/ultravista-Ks*/g_1.fits'.format(dirname))[0])
        matchimg = hdu[0].data
        hdu.close()

        hdu = fits.open(glob.glob('{}/ultravista-Ks*/deconv_01.fits'.format(dirname))[0])
        matchdec = hdu[0].data
        hdu.close()

        hdu = fits.open('./{}/a{}/watershed_segmaps/_id-{}.fits'.format(path, tile, _id))
        segmap = hdu[0].data
        segmap_coarse = img_as_bool(resize(segmap, (52, 52)))
        hdu.close()

        offsetcoords = []
        offsetcoords_dec = []
        tmplist = []
        skipdata = False
        for j, band in enumerate(bands[::-1]):
            hdu = fits.open(glob.glob('{}/{}*/g_1.fits'.format(dirname,band))[0])
            tmpimg = hdu[0].data
            hdu.close()

            hdu = fits.open(glob.glob('{}/{}*/deconv_01.fits'.format(dirname,band))[0])
            tmpdec = hdu[0].data
            hdu.close()

            if np.isnan(tmpimg).any():
                skipdata = True

            snr = getSNR(_id, j, snrs)
            if snr < 5:
                dy = 0.0
                dx = 0.0
            else:
                correlation = correlate(tmpimg, matchimg, mode='same', method='fft')
                tmpy, tmpx = np.unravel_index(np.argmax(correlation), correlation.shape)
                # maxy, maxx = np.unravel_index(np.argmax(tmpimg*segmap), tmpimg.shape)\

                dy = correlation.shape[0]/2.-tmpy
                dx = correlation.shape[0]/2.-tmpx

                if snr > 5 and np.sqrt(dy**2 + dx**2) > 3:
                    correlation = correlate(tmpimg*segmap_coarse, matchimg*segmap_coarse, mode='same', method='fft')
                    tmpy, tmpx = np.unravel_index(np.argmax(correlation*segmap_coarse), correlation.shape)

                    dy = (correlation.shape[0]/2.-tmpy)
                    dx = (correlation.shape[0]/2.-tmpx)

            tmpimg = np.roll(tmpimg, int(dy), 0)
            tmpimg = np.roll(tmpimg, int(dx), 1)
            if band in ['H', 'Y', 'J', 'zp', 'rp', 'V', 'B', 'Ks']:
                matchimg = tmpimg

            tmpdec = np.roll(tmpdec, 3*int(dy), 0)
            tmpdec = np.roll(tmpdec, 3*int(dx), 1)
            if snr < 5:
                dy = 0.0
                dx = 0.0
            else:
                correlation = correlate(tmpdec, matchdec, mode='same', method='fft')
                tmpy, tmpx = np.unravel_index(np.argmax(correlation*segmap), correlation.shape)

                dy_dec = (correlation.shape[0]/2.-tmpy)
                dx_dec = (correlation.shape[0]/2.-tmpx)

                # print (band, dy_dec, dx_dec, dirname)
            tmpdec = np.roll(tmpdec, int(dy_dec), 0)
            tmpdec = np.roll(tmpdec, int(dx_dec), 1)
            if band in ['H', 'Y', 'J', 'zp', 'rp', 'V', 'B', 'Ks']:
                matchdec = tmpdec


            if snr > 5 and np.sqrt(dy**2 + dx**2) > 5:
                print ('####################')
                print ('####################')
                print (band, dy, dx, dirname)
                plt.subplot(131)
                plt.imshow(tmpimg, origin='lower')
                plt.subplot(132)
                plt.imshow(matchimg, origin='lower')
                plt.subplot(133)
                plt.imshow(tmplist[j-1], origin='lower')
                plt.show()
                plt.close()
                print ('####################')
                print ('####################')
            offsetcoords.append([dy,dx])
            offsetcoords_dec.append([dy_dec,dx_dec])
        # if _id in [181078, 181952, 183184, 189746, 185187, 187119]:
        #     print (_id)
        #     print (np.array(offsetcoords)*3 + np.array(offsetcoords_dec))

        if skipdata:
            shutil.move(_dir, _dir.replace(filename, 'badphot/{}'.format(filename)))
        else:
            offsetcoords = np.array(offsetcoords)
            offsetTab = Table([offsetcoords[:,i] for i in range(2)], names=('dy', 'dx'), meta={'name': 'offset in pixels'})
            ascii.write(offsetTab, './{}/a{}/offsets/_id-{}.dat'.format(path, tile, int(_id)),\
                        overwrite=True, format='commented_header')

            offsetcoords_dec = np.array(offsetcoords_dec)
            offsetTab = Table([offsetcoords_dec[:,i] for i in range(2)], names=('dy', 'dx'), meta={'name': 'offset in pixels (deconvolved)'})
            ascii.write(offsetTab, './{}/a{}/offsets/_id-{}-dec.dat'.format(path, tile, int(_id)),\
                        overwrite=True, format='commented_header')

def return_offsets(catdir, catname, tile, path):

    table = ascii.read(catdir+catname) #cosmos_sfgcat_extended_update
    catalogs = np.array([table['z'], table['id'], table['x'], table['y'], table['ra'], table['dec']]).T
    ra = catalogs[:,4]
    dec = catalogs[:,5]

    if not os.path.isdir('./{}/a{}/offsets'.format(path, tile)):
        os.makedirs('./{}/a{}/offsets'.format(path, tile))

    hdu = fits.open('../images/{}/{}-ultravista_Ks.fits'.format(tile, tile))
    imgks = hdu[0].data
    header = hdu[0].header
    hdu.close()
    wcs = pywcs.WCS(header)

    xpos, ypos = wcs.all_world2pix(ra, dec, 1, ra_dec_order=True)

    tmpcat = np.loadtxt(catdir+catname, skiprows=1, usecols=[i for i in np.arange(11,43)])
    _ids = np.loadtxt(catdir+catname, skiprows=1, usecols=[0])
    _ids = list(_ids)
    snrs = 10**((tmpcat[:, ::2]-tmpcat[:, 1::2])/-2.5)

    def getSNR(_id, idx, snrcat):
        fluxes = [1,2,3,4,13,6,12,7,11,15,10,8,14] # making sure to retrieve the right flux
        id_idx = _ids.index(_id)
        return snrcat[id_idx, fluxes[idx]]


    for _dir in glob.glob('./{}/a{}/_id*'.format(path, tile))[:]:
        print (_dir)
        _id = int(os.path.basename(_dir).split('-')[1].split('_')[0])
        filename = os.path.basename(_dir)
        # print (_dir)

        idx = _ids.index(_id)
        x = xpos[idx]
        y = ypos[idx]


        matchimg = imgks[int(y-26):int(y+26), int(x-26):int(x+26)]

        oldlist = []
        tmplist = []
        offsetcoords = []

        skipdata = False
        for j, band in enumerate(bands[::-1]):
            hdu = fits.open('../images/{}/{}-{}.fits'.format(tile, tile, band))
            img = hdu[0].data
            hdu.close()
            tmpimg = img[int(y-26):int(y+26), int(x-26):int(x+26)]
            if np.isnan(tmpimg).any():
                # plt.imshow(tmpimg)
                # plt.show()
                skipdata = True
            oldlist.append(tmpimg)
    #         print (tmpimg.shape, matchimg.shape)

            snr = getSNR(_id, j, snrs)
            if snr < 5:
                dy = 0.0
                dx = 0.0
            else:
                correlation = correlate(tmpimg, matchimg, mode='same', method='fft')

    #             plt.imshow(correlation, origin='lower')
    #             plt.show()

                tmpy, tmpx = np.unravel_index(np.argmax(correlation), correlation.shape)
    #             print (correlation.shape, tmpy, tmpx)

                dy = correlation.shape[0]/2-tmpy
                dx = correlation.shape[0]/2-tmpx
                # print (band, dy, dx)

    #         matchimg = img[int(y-dy-26):int(y-dy+26), int(x-dx-26):int(x-dx+26)]
            matchimg = np.roll(tmpimg, int(dy), 0)
            matchimg = np.roll(tmpimg, int(dx), 1)
            tmplist.append(matchimg)
    #         if snr > 5 and np.sqrt(dy**2 + dx**2) > 5:
    #         if int(_id) == 114072:
    #             print (band, dy, dx)
    #             plt.subplot(131)
    #             plt.imshow(tmpimg, origin='lower')
    #             plt.subplot(132)
    #             plt.imshow(matchimg, origin='lower')
    #             plt.subplot(133)
    #             plt.imshow(tmplist[j-1], origin='lower')
    #             plt.show()

            if y+dy>4070 or x+dx>4070:
                print ('####################')
                print ('####################')
                print (y-dy, x-dx)
                print ('####################')
                print ('####################')
            offsetcoords.append([dy,dx])

        if skipdata:
            shutil.move(_dir, _dir.replace(filename, 'badphot/{}'.format(filename)))
        else:
            offsetcoords = np.array(offsetcoords)
            offsetTab = Table([offsetcoords[:,i] for i in range(2)], names=('dy', 'dx'), meta={'name': 'offsef in pixels'})
            ascii.write(offsetTab, './{}/a{}/offsets/_id-{}.dat'.format(path, tile, int(_id)),\
                        overwrite=True, format='commented_header')


    #     oldb = oldlist[9]
    #     oldz = oldlist[4]
    #     oldk = oldlist[0]
    #     oldrgb = create_rgb(oldb, oldz, oldk, [np.median(oldb), np.median(oldz), np.median(oldk)])

    #     b = tmplist[9]
    #     z = tmplist[4]
    #     k = tmplist[0]
    #     rgbimg = create_rgb(b, z, k, [np.median(b), np.median(z), np.median(k)])

    #     print ('../run/images/{}/id{}-offsets.dat'.format(tile, int(_id)))

    #     plt.subplots(1,2)
    #     plt.subplot(121)
    #     plt.imshow(oldrgb, origin='lower')

    #     plt.subplot(122)
    #     plt.imshow(rgbimg, origin='lower')
    #     plt.show()
