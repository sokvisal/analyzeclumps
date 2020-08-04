import string
import numpy as np
import astropy.io.fits as pyfits
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

def write_vars_to_file(_f, **vars):
    for (name, val) in vars.items():
        _f.write("%s = %s\n" % (name, repr(val)))

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gauss(data):
    data = data/sum(data)

    x = np.arange(0,len(data))

    n = len(x)                          #the number of data
    mean = sum(x*data)/n                   #note this correction
    sigma = sum(data*(x-mean)**2)/n       #note this correction

    popt,pcov = curve_fit(gaus,x,data,p0= [1., mean, sigma])

    return [popt, pcov]

def check_decon(directory='results/',save=None, HST=None):
    hdulist = pyfits.open(directory+'deconv_01.fits')[0]
    deconvolve = hdulist.data

    hdulist = pyfits.open(directory+'g_1.fits')[0]
    data = hdulist.data

    hdulist = pyfits.open(directory+'resi_01.fits')[0]
    resi = hdulist.data

    newdata = [data, deconvolve, np.abs(resi/data)]
    title = ['Original', 'Deconvolved', 'Relative Difference']
    if HST:
        hdulist = pyfits.open(directory+save.replace('.pdf', '')+'.fits')[0]
        hst = hdulist.data[125:375, 125:375]
        print (hst.shape)

        newdata = [data, deconvolve, hst]
        title = ['Original', 'Deconvolved', 'HST']

    plt.subplots(1,3, figsize=(12, 4))
    for i, data in enumerate(newdata):
        plt.subplot(1,3,i+1)
        if i == 2 and not HST:
            plt.imshow(data, origin='lower', vmin=0, vmax=1)
        elif i == 2 and HST:
            plt.imshow(data, cmap=plt.get_cmap('Greys'), origin='lower', vmax=0.1)
        else:
            #plt.contour(data, origin='lower', levels=np.linspace(np.min(data), np.max(data), 4), linewidths=1.5)
            plt.imshow(data,cmap=plt.get_cmap('Greys'),origin='lower')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(title[i])
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save:
        print ('saving plot')
        plt.savefig(save)
    plt.show()

def norm_data(data, hst_data):
    return (data - np.min(data))*((np.max(hst_data)-np.min(hst_data))/(np.max(data)-np.min(data)))+np.min(hst_data)

def find_corrections(gdir=None, hstdir=None):
    import pymcs.src.lib.pyssep.src.findstars as findstars
    import numpy as np
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    #TODO: change directories and search more directories
    gdir = '../images/CFHTLS_2482_15946.fits'
    hstdir = '150.50188000_2.51953000_hst.fits'

    #os.chdir('../images/')
    cfhtls = findstars.get_objects(gdir)
    cfhtls_x, cfhtls_y = cfhtls[0], cfhtls[1]
    hdulist = pyfits.open(gdir)[0]
    cfhtls_h = hdulist.header

    w = pywcs.WCS(cfhtls_h)
    cfhtls_ra, cfhtls_dec = w.all_pix2world(cfhtls_x, cfhtls_y, 1, ra_dec_order=True)

    # find stars in hst field
    hst = findstars.get_objects(hstdir)
    hst_sc = hst[6]
    idx = np.where(hst_sc>0.8)
    hst_x = hst[0][idx]
    hst_y = hst[1][idx]

    hdulist = pyfits.open(hstdir)[0]
    hst_d = hdulist.data
    hst_h = hdulist.header

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,12))
    plt.scatter(hst_x,hst_y, color='r', s=5, marker='x')
    plt.imshow(hst_d, origin='lower', vmax=15, cmap=plt.get_cmap('Greys'))
    plt.colorbar()
    plt.show()

    w = pywcs.WCS(hst_h)
    hst_ra, hst_dec = w.all_pix2world(hst_x, hst_y, 1, ra_dec_order=True)

    # find closest coordinates to hst stars
    adj = []
    for i,j in zip(hst_ra, hst_dec):
        tmp = (np.abs(i-cfhtls_ra)).argmin()
        tmp2 = (np.abs(j-cfhtls_dec)).argmin()
        adj.append([i-cfhtls_ra[tmp], j-cfhtls_dec[tmp2] ])

    adj = np.array(adj)
    tmp = np.where(np.diff(adj, axis=0)[:,0]>0.001)[0]
    adj = np.delete(adj, tmp, axis=0).mean(axis=0)
    return adj

# function to calculate the residual of two dataset with different normalization
def get_res(data1, data2, variance):
    sum1 = np.sum(data1)
    sum2 = np.sum(data2)

    fact = sum1/sum2
    data1 = data1/fact

    res = np.abs(data1 - data2)
    xsq = np.sum(((data1 - data2)**2.)/variance)
    mse = (((data1 - data2)**2.).mean())
    psnr = 10.*np.log10(np.max(data1)**2/mse)
    return res, round(xsq, 4), round(psnr, 2), fact

def get_fact(header1, shape1, header2, shape2):
    ### the data for header1 is more resolved

    import astropy.wcs as pywcs

    w1 = pywcs.WCS(header1)
    #
    # x1, y1 = w1.all_pix2world(np.arange(0,shape1), np.arange(0,shape1), 1, ra_dec_order=True)
    # #d1 = header1['CDELT1']
    # dx1 = np.mean(np.diff(x1))
    # dy1 = np.mean(np.diff(x1))
    #
    # w2 = pywcs.WCS(header2)
    # #d2 = header2['CDELT1']
    # x2, y2 = w2.all_pix2world(np.arange(0,shape1), np.arange(0,shape1), 1, ra_dec_order=True)
    # #d1 = header1['CDELT1']
    # dx2 = np.mean(np.diff(x2))
    # dy2 = np.mean(np.diff(x2))

    d2 = 1.388889e-5 #pixelscale of groundbased
    try: d1 = abs(header1['CD1_1'])
    except: d1 = abs(header1['CDELT1'])
    # print (d1, d2)

    return d2/d1

def gaussian2D(shape, fwhm, c1, c2, i0, fwhm0 = 0.):
        """
        Return a gaussian
        Inputs:
         - shape
         - fwhm
         - c1
         - c2
         - i0
         - fwhm0: if this value is different from zero, the function will use it to simulate
                  a convolution. In practice it will add it to the basic FWHM of the gaussians
                  (default = 0.)
        """
        indexes = np.indices(shape)
        k = 2. * np.sqrt(2.*np.log(2.))
        sig_2 =  (fwhm  / k)**2.
        sig0_2 = (fwhm0 / k)**2.
        norm = sig_2/(sig_2+sig0_2)
        g = np.exp((-(indexes[0]-c1+0.5)**2. -
                  (indexes[1]-c2+0.5)**2)/(2.*(sig_2+sig0_2)))

        #if fwhm<1: return i0*g*norm*(1./(2*np.pi*sig_2))/np.sum(i0*g*norm*(1./(2*np.pi*sig_2)))
        return i0*g*norm*(1./(2*np.pi*sig_2))

def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def return_hst(hst_header, hstData, im_header, imShape, corr):
    import astropy.wcs as pywcs

    # hst header
    whst = pywcs.WCS(hst_header)
    hstx, hsty = whst.all_pix2world(np.arange(0,hstData.shape[0]), np.arange(0,hstData.shape[0]), 1, ra_dec_order=True)

    # image header
    wim = pywcs.WCS(im_header)
    tmp = wim.all_pix2world(0, 0, 1, ra_dec_order=True)
    x2, y2 = wim.all_pix2world(np.arange(0,imShape), np.arange(0,imShape), 1, ra_dec_order=True)

    xidx = (np.abs(hstx - x2[0])).argmin()
    yidx = (np.abs(hsty - y2[0])).argmin()

    xidx2 = (np.abs(hstx - x2[-1])).argmin()
    yidx2 = (np.abs(hsty - y2[-1])).argmin()

    if corr is None:
        return hstData[yidx:yidx+imShape, xidx:xidx+imShape]
    else:
        return hstData[yidx+corr[1]:yidx+corr[1]+imShape, xidx+corr[0]:xidx+corr[0]+imShape]

def inpltzoom(fact, data, header=None):
    from scipy.ndimage.interpolation import zoom

    zdata = zoom(data, np.abs(fact))
    zdata = zdata*(np.sum(data)/np.sum(zdata))

    if header:
        zheader = adjustHeader(header=header, newcenter=zdata.shape[0]/2., sampleFact=fact, corr=None)
        return zdata, zheader
    else:
        return zdata

def get_offset(hst, refimage, refimage2):
    from scipy.signal import correlate

    correlation = correlate(hst, refimage, method='fft')
    tmpy, tmpx = np.unravel_index(np.argmax(correlation), correlation.shape)

    correlation2 = correlate(hst, refimage2, method='fft')
    tmpy2, tmpx2 = np.unravel_index(np.argmax(correlation), correlation.shape)

    y = np.mean([tmpy, tmpy2])
    x = np.mean([tmpx, tmpx2])

    import matplotlib.pyplot as plt

    # plt.subplot(1,3,1)
    # plt.imshow(refimage, origin='lower')
    # plt.subplot(1,3,2)
    # plt.imshow(hst[c-itv:c+itv,c-itv:c+itv], origin='lower')
    # plt.subplot(1,3,3)
    # plt.imshow(result, origin='lower')
    # plt.show()

    from scipy.ndimage.measurements import center_of_mass
    # y, x = center_of_mass(correl)
    # y, x  = np.unravel_index(np.argmax(result), result.shape)
    dy, dx = y-correlation.shape[0]/2., x-correlation.shape[0]/2.
    return dy, dx

def cutout(hstdata, hstheader, galaxydata, deconvdata, deconvheader, band, kern = None, correlate = False):
    import astropy.wcs as pywcs
    from scipy.interpolate import RectBivariateSpline, interp2d
    import matplotlib.pyplot as plt

    from scipy.signal import correlate

    def get_pixscale(header):
        '''
        Return pixel scale in arcsec
        '''
        return (np.abs(header['CD1_1'])+np.abs(header['CD2_2']))/2 * 3600.

    whst = pywcs.WCS(hstheader)

    # image header
    wim = pywcs.WCS(deconvheader)
    imra, imdec = wim.all_pix2world(deconvdata.shape[0]/2,deconvdata.shape[0]/2, 1)

    x, y = whst.all_world2pix(imra, imdec, 1)

    if correlate:
        pixelscale = [1.3888888888888e-5, 8.333333e-6, 2.777777777777e-5, 1.388889e-5] # pixel scale of b,g - z - h - ground
        imsize = 7.8
        tmpimsize = 10.0

        if band == 'b':
            l = imsize/(pixelscale[0]*3600)
            tmpl = tmpimsize/(pixelscale[0]*3600)
            tlen = int(imsize/(pixelscale[0]*3600))
            sfact = pixelscale[0]/pixelscale[1]
        elif band == 'z'  or band == 'v':
            l = imsize/(pixelscale[1]*3600)
            tmpl = tmpimsize/(pixelscale[1]*3600)
            tlen = int(imsize/(pixelscale[1]*3600))
            sfact = pixelscale[1]/pixelscale[1]
        elif band == 'h':
            l = imsize/(pixelscale[2]*3600)
            tmpl = tmpimsize/(pixelscale[2]*3600)
            tlen = int(imsize/(pixelscale[2]*3600))
            sfact = pixelscale[2]/pixelscale[1]

        tmpimg = hstdata[int(y)-int(tmpl/2):int(y)+int(tmpl/2), int(x)-int(tmpl/2):int(x)+int(tmpl/2)]
        tmpx = np.arange(0, tmpimg.shape[0])
        tmpy = np.arange(0, tmpimg.shape[1])

        # print band, len(tmpx), len(tmpy), tmpimg.shape
        samp = RectBivariateSpline(tmpx, tmpy, tmpimg)

        cx, cy = [tmpx.shape[0]/2+int(x)-x, tmpx.shape[0]/2+int(y)-y]
        nx = np.linspace(cx-l/2., cx+l/2., tlen)
        ny = np.linspace(cy-l/2., cy+l/2., tlen) #tlen

        orimg = samp(ny, nx)

        dy, dx = get_offset(orimg, galaxydata, deconvdata)
        # print x, y, cx, cy
        # print 'dy = {}, dx = {}'.format(str(dy), str(dx))

        cx, cy = [tmpx.shape[0]/2+int(x)-x+dx, tmpx.shape[0]/2+int(y)-y+dy]
        nx = np.linspace(cx-l/2., cx+l/2., tlen)
        ny = np.linspace(cy-l/2., cy+l/2., tlen) #tlen
        iniflux = np.sum(samp(ny, nx))

        tlen = int(imsize/(pixelscale[1]*3600))
        cx, cy = [tmpx.shape[0]/2+int(x)-x+dx, tmpx.shape[0]/2+int(y)-y+dy]
        # note that RectBivariateSpline flip the y and x axis
        nx = np.linspace(cx-l/2., cx+l/2., tlen)
        ny = np.linspace(cy-l/2., cy+l/2., tlen)

        zimg = samp(ny, nx)#/sfact**2
        zimg *= iniflux/np.sum(zimg)

        # plt.subplot(1,3,1)
        # plt.imshow(deconvdata, origin='lower', cmap='Greys')
        #
        # plt.subplot(1,3,2)
        # plt.imshow(orimg, origin='lower', cmap='Greys')
        # plt.title('Band = {}'.format(band))
        #
        # plt.subplot(1,3,3)
        # plt.imshow(zimg, origin='lower', cmap='Greys') #[zimg.shape[0]/2-30:zimg.shape[0]/2+30, zimg.shape[0]/2-30:zimg.shape[0]/2+30]
        # plt.show()

        return zimg #inpltzoom(sfact, zimg)
    else:
        if corr:
            x, y = x+corr[0], y+corr[1]
        else:
            pass


        # cutouthst = hstdata[y:y+deconvdata.shape[0], x:x+deconvdata.shape[0]]
        cutouthst = hstdata[int(y)-deconvdata.shape[0]/2:int(y)+deconvdata.shape[0]/2, int(x)-deconvdata.shape[0]/2:int(x)+deconvdata.shape[0]/2]

        from astropy.nddata.utils import Cutout2D

        hdu_crop = Cutout2D(hstdata, (x, y), cutouthst.shape[0], wcs=pywcs.WCS(hstheader))
        # Cropped WCS
        wcs_cropped = hdu_crop.wcs
        # Update WCS in header
        from copy import deepcopy
        tmpheader = deepcopy(hstheader)
        tmpheader.update(wcs_cropped.to_header())

        return cutouthst, tmpheader

def adjustHeader(header, newcenter, sampleFact, corr):
    ## Adjusting header for resampling data ##

    header['CRPIX1'] = newcenter
    header['CRPIX2'] = newcenter

    # header['CRVAL1'] += 6.6388889e-5#4.416667e-5
    # header['CRVAL2'] -= 6.416667e-5#5.638889e-5

    header['PC1_1'] = header['PC1_1']/sampleFact
    header['PC2_2'] = header['PC2_2']/sampleFact
    if corr == None:

        return header
    else:
        # print corr

        # header['CRVAL1'] += corr[0]
        # header['CRVAL2'] -= corr[1]
        return header

def conv(res, hstData):
    ### ratio is the sfact/gres
    from scipy.signal import convolve2d

    fact = res#/0.03 # factor of pscale and hst pixel scale

    r_len = 32#int(res*3)
    c1, c2 =  r_len/2.-0.5, r_len/2.-0.5 #-0.5 to align on the pixels grid

    gres = fact
    kern = gaussian2D((r_len,r_len), gres, c1, c2, 1.)

    chst = convolve2d(hstData, kern, mode='same')
    return chst

def cmdist(z):
    c = 2.99e5
    ho = 67.8

    om = 0.308
    olam = 0.792

    dh = c/ho

    return dh/(np.sqrt(om*(1+z)**3 + olam))

def fwhm(hst=.1/0.03, conv=2.5):
    return (np.sqrt((2.35*hst)**2. + (2.35*conv)**2)/2.35)/2.

def snr(so, ss, n, cfact):
    q = 0.59 # for quantum efficiency i band
    ss = 2.668*cfact # avg background counts per pixels
    t = 74723.95 # exposure times
    c = 0.51 # conversion to electron per seconds
    # print (so*np.sqrt(t)*c)/np.sqrt(so + ss*n)
    #return so*np.sqrt(q*t)/np.sqrt(so + ss*n) # in units of adu
    # print 'snr:', (so)/np.sqrt(so + ss)
    return (so)/np.sqrt(so + ss*n)

def create_mask(data):
    # create mask using sextractor
    from pymcs.src.lib import pysex

    # plt.imshow(data)
    # plt.show()

    cat = pysex.run(data, params=['X_IMAGE', 'Y_IMAGE', 'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'KRON_RADIUS', 'MU_MAX', 'ELONGATION'])
    x, y, si, mu, elong = cat['X_IMAGE'].tonumpy(), cat['Y_IMAGE'].tonumpy(), cat['SPHEROID_SERSICN'].tonumpy(), cat['MU_MAX'].tonumpy() , cat['ELONGATION'].tonumpy()

    shape = data.shape
    cx, cy = shape[0]/2.,shape[0]/2.
    idx = np.argmin(np.abs(x-cx) + np.abs(y-cy)/2.)

    theta = cat['THETA_IMAGE'].tonumpy()[idx]
    r = cat['KRON_RADIUS'].tonumpy()[idx]
    a, b = cat['A_IMAGE'].tonumpy()[idx]*r, cat['B_IMAGE'].tonumpy()[idx]*r

    indexes = np.indices(shape)
    # general equation of an ellipse
    g = (((indexes[0]-cx)*np.cos(theta)-(indexes[1]-cy)*np.sin(theta))/a)**2 + (((indexes[0]-cx)*np.sin(theta)+(indexes[1]-cy)*np.cos(theta))/b)**2
    # mask
    g[np.where(g<1)]=1
    g[np.where(g>1)]=0

    tmp = np.argwhere(g)
    (ystart, xstart), (ystop, xstop) = tmp.min(0), tmp.max(0) + 1

    mask = g[ystart:ystop, xstart:xstop]

    return mask, [ystart, ystop, xstart, xstop], mu[idx], elong[idx]

def getHSTFact(hstdirectory):
    ps = [1.38888888e-05, 8.33333338e-6, 2.7777777e-5]

    if 'ultravista-h' in hstdirectory:
        return ps[2]/ps[1]
    elif 'subaru-b' in hstdirectory or 'subaru-v' in hstdirectory:
        return ps[0]/ps[1]
    else:
        return ps[1]/ps[1]

def openFits(directory, hduid=0):
    from astropy.io import fits
    hdu = fits.open(directory)[hduid]
    return hdu.data, hdu.header


def getnoise(directory):
    '''
    Note that this aperture is selected for T065 tile of HST,
    Need to update this noise selection for other tiles and make it more automatic
    '''

    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt

    data = fits.open('../images/'+directory)[0].data

    def createCircularMask(h, w, center=None, radius=None):

        if center is None: # use the middle of the image
            center = [int(w/2), int(h/2)]
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

    coords = np.array([[967, 2428], [1077, 2425], [1555,3497], [2639, 3642], [3428, 3300], [3511, 3268], [3534, 2065], [3578, 2083], \
                       [3692, 2176], [3704, 2122], [3703, 2122], [3593, 2044], [3309, 1224], [3455, 1178], [3532, 1281], [3435, 1319], \
                       [2828, 967], [2928, 991]])

    r = 4
    f = 1#0.8063

    noise = []
    for i in coords:
        tmp = data[i[0]*f-r:i[0]*f+r, i[1]*f-r:i[1]*f+r]
        h, w = tmp.shape[:2]
        mask = createCircularMask(h, w)
        masked_img = tmp.copy()
        masked_img[~mask] = 0

        noise.append( np.sum(masked_img) )

    a = len(np.where(masked_img>0)[0])
    return np.array(noise).var()/(a*9.)

class FindHSTCounter():

    def save_HST_img(self, deconv_directory, hst_directory, hduid):
        from imutil import rot
        import os

        deconvolve, deconvolve_h = openFits(deconv_directory+'/deconv_01.fits', hduid=0)
        galaxy, galaxy_h = openFits(deconv_directory+'/g_1.fits', hduid=0)
        hst, hst_h = openFits(hst_directory, hduid=hduid)
        if hduid:
            print ('what')
            hst, hst_h = rot(hst, hst_h, galaxy_h)


        # find sampling factor for deconv data
        self.fact = get_fact(hst_h, hst.shape[0], deconvolve_h, deconvolve.shape[0])
        # self.fact = get_fact(rothst_h, rothst.shape[0], deconvolve_h, deconvolve.shape[0])
        self.s_fact = float(search_cfg(deconv_directory+'/config.py', 'S_FACT'))
        self.g_res = float(search_cfg(deconv_directory+'/config.py', 'G_RES'))
        imagedir = search_cfg(deconv_directory+'/config.py', 'FILENAME')[1:-1]

        # interpolated galaxy to hst pixel density
        gal = inpltzoom(self.s_fact, galaxy)
        zgal = inpltzoom(self.fact, gal)

        zdeconv, zheader = inpltzoom(self.fact, deconvolve, deconvolve_h)

        tmp = os.path.basename(deconv_directory)
        if 'ultravista-H' in deconv_directory:
            kerndir = False
            hst_savedir = deconv_directory.replace(tmp, '')+'f160w_ps.fits'
            band = 'h'
        elif 'subaru-zp' in deconv_directory:
            kerndir = '/mnt/drivea/run/kernpsf/F814W_kern.fits'
            hst_savedir = deconv_directory.replace(tmp, '')+'f814w_ps.fits'
            band = 'z'
        elif 'subaru-V' in deconv_directory:
            kerndir = '/mnt/drivea/run/kernpsf/F606W_kern.fits'
            hst_savedir = deconv_directory.replace(tmp, '')+'f606w_ps.fits'
            band = 'v'
        else:
            kerndir = '/mnt/drivea/run/kernpsf/F475W_kern.fits'
            hst_savedir = deconv_directory.replace(tmp, '')+'f475w_ps.fits'
            band = 'b'

        # load PSF kernel
        if kerndir:
            kern, kernh = openFits(kerndir, hduid=0)
            self.hst = cutout(hst, hst_h, zgal, zdeconv, zheader, band = band, kern = kern, correlate = True)
            # self.hst = cutout(hst, hst_h, zdeconv, zheader, band = band, kern = kern, correlate = True)
        else:
            self.hst = cutout(hst, hst_h, zgal, zdeconv, zheader, band = band, correlate = True)
            # self.hst = cutout(hst, hst_h, zdeconv, zheader, band = band, correlate = True)

        # # interpolated deconv to hst pixel density
        # zdeconv, zheader = self.__return_intrpl(self.fact, deconvolve, deconvolve_h)
        #
        # inihst, tmp = cutout(hst, hst_h, zdeconv, zheader, corr=None)
        #
        # y, x = self.__cor_offset(inihst, zdeconv, std) # find astrometric offsets
        # self.hst, self.hst_h = cutout(hst, hst_h, zdeconv, zheader, corr=[int(x),int(y)])

        # saving hst imaging
        hdu = pyfits.PrimaryHDU(self.hst) #, header=self.hst_h)
        hdu.writeto(hst_savedir, clobber=True)

    def findHST(self, directory):
        import os
        #######################################################################################################
        self.save = None#save
        self.bband_dir = directory + '/subaru-B' #subaru-B_lambd-000100.0
        self.gband_dir = directory + '/subaru-V'
        self.zband_dir = directory + '/subaru-zp'
        self.jband_dir = directory + '/ultravista-H'

        # def convHST(self):
        from scipy.ndimage.interpolation import zoom
        from pymcs.src.lib import pysex
        import astropy.io.fits as pyfits
        import glob

        directories = [self.bband_dir, self.gband_dir, self.zband_dir, self.jband_dir]
        HSTdirectories = ['/mnt/drivea/run/f475w/test2.fits', '/mnt/drivea/run/f606w/test2.fits', '/mnt/drivea/run/f814w/t065.fits', '/mnt/drivea/run/f160w/t065.fits']
        HSTdirectories = ['/mnt/drivea/run/f475w/test2.fits', '/home/visal/Downloads/hlsp_candels_hst_acs_cos-tot_f606w_v1.0_drz.fits', '/mnt/drivea/run/f814w/t065.fits', '/mnt/drivea/run/f160w/t065.fits']
        #'../images/data_hst.fits', '../images/hst_t065_h.fits']
        # ['../run/hst_f475w/j8pu3q010_drc.fits', '../images/data_hst.fits', '../images/hst_t065_h.fits']

        directories = [self.jband_dir]
        HSTdirectories = ['/mnt/drivea/run/f160w/t065.fits']
        for i, drt in enumerate(directories):
            if os.path.isdir(drt):
                self.save_HST_img(directories[i], HSTdirectories[i], hduid=0)
            else:
                # print drt
                pass


class checkResidual():
    def __return_intrpl(self, fact, data, header=None):
        from scipy.ndimage.interpolation import zoom

        zdata = zoom(data, np.abs(fact))
        zdata = zdata*(np.sum(data)/np.sum(zdata))

        if header:
            zheader = adjustHeader(header=header, newcenter=zdata.shape[0]/2., sampleFact=fact, corr=None)
            return zdata, zheader
        else:
            return zdata

    def __cor_offset(self, hst, deconv, std):

        # import stsci
        from scipy.signal import convolve2d
        from scipy.ndimage import correlate

        c = int(deconv.shape[0]/2.)
        itv = int(c/3.)

        # zdeconv = zdeconv/(np.sum(zdeconv)/np.sum(inihst))
        # correl = convolve2d(zdeconv.T, inihst, mode='same')/std**2
        correl = correlate(deconv[c-itv:c+itv,c-itv:c+itv], hst[c-itv:c+itv,c-itv:c+itv], mode='constant')/std**2 #[c-itv:c+itv,c-itv:c+itv]

        # import matplotlib.pyplot as plt
        # plt.imshow(correl)
        # plt.show()
        y, x  = np.unravel_index(np.argmax(correl), correl.shape)
        y, x = correl.shape[0]/2.-y, correl.shape[0]/2.-x
        return y, x

    def __fit_res(self, model, data):
        '''
        model   - this is the array that will be convolved with a Gaussian
        data    - this is the array that will act the ground "truth"
        '''
        def rmsFunc(shape, fwhm):
            return conv(res=fwhm, hstData=model).ravel()

        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(rmsFunc, model.shape, data.ravel(), p0=25)#, sigma=(1./np.sqrt(np.abs(data)+std**(2.))).ravel())
        fwhm, dfwhm = popt[0], np.sqrt(np.diag(pcov))[0]
        return fwhm, dfwhm

    def __tmp(self, data, savedir):
        from pymcs.src.lib import pysex

        drt = '../'+savedir+'/segmap.fits'
        cat = pysex.run(data, params=['X_IMAGE', 'Y_IMAGE'], conf_args={'CHECKIMAGE_TYPE':'SEGMENTATION', 'CHECKIMAGE_NAME': drt})


    def resiDec(self, directory, band):

        #######################################################################################################
        self.save = None#save
        self.directory = directory + '/lambd*'
        self.hst_directory = '../images/data_'#hst_directory

    #def convHST(self):
        from scipy.ndimage.interpolation import zoom
        from pymcs.src import _1_get_sky
        from pymcs.src.lib import pysex
        import astropy.io.fits as pyfits
        import glob

        #directory = self.tmpDir
        hst_directory = self.hst_directory
        directory = self.directory

        directory = glob.glob(directory)[0]

        deconvolve, deconvolve_h = openFits(directory+'/deconv_01.fits')    # deconv data
        galaxy, galaxy_h = openFits(directory+'/g_1.fits')                  # galaxy
        if band == 'z':
            hst, hst_h = openFits(hst_directory+'hst.fits')                 # hst i-band
        else:
            hst, hst_h = openFits('../images/hst_t065_h.fits')              # hst h-band

        # find sampling factor for deconv data
        self.fact = get_fact(hst_h, hst.shape[0], deconvolve_h, deconvolve.shape[0])
        self.s_fact = float(search_cfg(directory+'/config.py', 'S_FACT'))
        self.g_res = float(search_cfg(directory+'/config.py', 'G_RES'))
        imagedir = search_cfg(directory+'/config.py', 'FILENAME')[1:-1]

        # interpolated galaxy to hst pixel density
        gal = self.__return_intrpl(self.s_fact, galaxy)
        zgal = self.__return_intrpl(self.fact, gal)
        # zgal = zoom(gal, self.fact)

        c, std = _1_get_sky.get_sky_val(galaxy, show=False, range=[0.01,0.99], nbins=20, save=None)

        # interpolated deconv to hst pixel density
        zdeconv, zheader = self.__return_intrpl(self.fact, deconvolve, deconvolve_h)
        inihst, tmp = cutout(hst, hst_h, zdeconv, zheader, corr=None)

        import matplotlib.pyplot as plt

        # plt.imshow(zdeconv)
        # plt.show()
        #
        # plt.imshow(inihst)
        # plt.show()

        y, x = self.__cor_offset(inihst, zdeconv, std) # find astrometric offsets
        self.hst, self.hst_h = cutout(hst, hst_h, zdeconv, zheader, corr=[int(x),int(y)])
        hst = self.hst

        ###########################################################
        ############# saving hst imaging ##########################
        ###########################################################
        # if band == 'z':
        hdu = pyfits.PrimaryHDU(self.hst, header=self.hst_h)
        hdu.writeto(self.directory.replace('/lambd*','')+'/hst.fits', clobber=True)
        ###########################################################
        ###########################################################

        tmpfact = np.sum(galaxy)/np.sum(zgal)
        zgal = zgal*tmpfact

        ############# create mask based on sextractor ################
        self.mask, self.trim_arr, mu, elong = create_mask(zgal)
        # print mu, elong
        htrim = hst[self.trim_arr[0]:self.trim_arr[1], self.trim_arr[2]:self.trim_arr[3]]
        gtrim = zgal[self.trim_arr[0]:self.trim_arr[1], self.trim_arr[2]:self.trim_arr[3]]

        masked_gal = self.mask*gtrim
        self.masked_hst = self.mask*htrim
        masked_hst = self.masked_hst
        ##############################################################


        frac = np.sum(masked_hst)/np.sum(masked_gal)
        masked_gal = frac*masked_gal

        self.snr = snr(so = np.sum(masked_gal), ss=std**2, n=np.sum(self.mask), cfact=tmpfact) #galaper(hst, zgal, True)

        fwhm, dfwhm = self.__fit_res(self.masked_hst, masked_gal)
        fwhm = np.sqrt(0.1**2 + (fwhm*0.03)**2)

        self.chsto = conv(res=fwhm, hstData=self.hst) # hst convolved to original image
        # print np.max(masked_gal), np.max(self.chsto)

        self.tmpData = []
        self.tmpValues = []

        self.gal = gal

        res, mse, psnr, fact = get_res(zgal, self.chsto, std**2)
        # print fwhm, rms, mse

        newdata = np.array([self.hst, self.chsto, zgal/fact, res])
        self.tmpData.append(newdata)
        self.tmpValues.append(['galaxy', self.snr, mse, fwhm])
        galfwhm = fwhm

        #################################################################################################################

        import glob
        from tqdm import tqdm
        import os

        #directory = self.tmpDir
        directory = glob.glob(self.directory)
        masked_hst = self.masked_hst
        hst = self.hst
        gal = self.gal

        self.lambd_test = []
        for i, fn in enumerate(directory):
            import os
            tmp =  os.path.dirname(fn).split('_')
            tmp2 =  os.path.basename(fn).split('_')
            ra, dec, z, lambd = float(tmp[4]), float(tmp[5]), float(tmp[7]), float(tmp2[1])
            self.masterDict = {'ra': ra, 'dec': dec, 'z':z, 'lambda': lambd}

            ###### Deconvolved #######
            deconvolve, deconvolve_h = openFits(fn+'/deconv_01.fits')
            if band == 'h':
                # self.__tmp(galaxy, directory[0]) # segmap
                # segmap = pyfits.open(directory[0]+'/segmap.fits')[0].data
                # self.segmap = segmap.repeat(3, axis = 0).repeat(3, axis = 1).T
                # self.segmap[self.segmap > 1] = 1

                correlnoise = getnoise(imagedir)
                makeVBin(fn, deconvolve, correlnoise, None)

            # interpolated deconv to hst pixel density
            zdeconv = zoom(deconvolve, np.abs(self.fact)) # resample to match hst sampling
            zdeconv = zdeconv*(np.sum(deconvolve)/np.sum(zdeconv))
            c, std = _1_get_sky.get_sky_val(zdeconv, show=False, range=[0.1,0.90], nbins=55, save=None)

            masked_dec = zdeconv[self.trim_arr[0]:self.trim_arr[1], self.trim_arr[2]:self.trim_arr[3]]#self.mask.cutout(zdeconv)

            frac = np.sum(masked_hst)/np.sum(masked_dec)
            masked_dec = frac*masked_dec

            # popttmp, pcovtmp = curve_fit(rms_gal, gal.shape, gal.ravel(), p0=21 )
            fwhm, dfwhm = self.__fit_res(self.masked_hst, masked_dec)
            self.chst = conv(res=fwhm, hstData=self.hst)

            fwhm = np.sqrt(0.1**2 + (fwhm*(0.03))**2)
            res, mse, psnr, fact = get_res(zdeconv, self.chst, std**2)
            hdu = pyfits.PrimaryHDU(res, header=self.hst_h)
            hdu.writeto(fn+'/hst_resi.fits', clobber=True)
            # print self.masked_hst.shape, self.chst.shape, (masked_dec/fact).shape, res.shape
            #print np.max(zgal/fact), np.max(chsto)

            self.masterDict.update({'snr':self.snr, 'sfact':self.s_fact, 'gres':self.g_res, 'psnr':psnr})
            self.lambd_test.append([lambd, fwhm, mse, galfwhm, self.snr,  mu, elong])#({'lambda': lambd, 'fwhm':fwhm, 'aper_rms':aper_rms})

            newdata = np.array([self.hst, self.chst, zdeconv/fact, res])
            self.tmpData.append(newdata)
            self.tmpValues.append([fn, psnr, mse, fwhm])

        # self.resiDec()
        np.save(self.directory.replace('/lambd*','') + '/dict.npy', self.lambd_test)
        #self.stackDec()
        pdf = PDF()
        name = os.path.basename(self.directory.replace('/lambd*', ''))
        # print name
        pdf.plots(directory=self.directory.replace('/lambd*',''), name=name, tmpData=self.tmpData, tmpValues=self.tmpValues)

    def stackDec(self):
        data = np.array(self.tmpData)[1:,2] # all deconvolved images
        stacked = data.mean(axis=0)

        chst = self.chst
        hst = self.hst

        dres, dmse, dpsnr, dfact = get_res(stacked, hst)

        newdata = np.array([hst, chst, stacked, dres])
        self.tmpData.insert(1,newdata)
        self.tmpValues.insert(1,['Stacked Deconvolved', dpsnr, dmse])

class photometry():
    def __init__(self, z, origin, deconvolved, hst):
        self.z = z
        self.origin = origin
        self.deconvolved = deconvolved
        self.hst = hst

    def difference(self):
        z = self.z

        def frac_ee(pixel):
            from scipy.interpolate import interp1d

            eec = [0.322, 0.611, 0.770, 0.830, 0.853, 0.871, 0.889, 0.901, 0.908, 0.914, 0.949, 0.972]
            pix = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 20., 40.]

            f2 = interp1d(pix, eec, kind='cubic')

            return f2(pixel)

        from photutils import aperture_photometry
        from photutils import CircularAperture
        import scipy.integrate as integrate

        center = [self.origin.shape[0]/2.,self.origin.shape[0]/2.]
        zpst = -2.5*np.log10(1.271e-19) -21.10
        cfht_zp = 2.5*np.log10(2.346615770e+05) + 25.774 - 0.04*(0.040-1)
        #print cfht_zp

        radi = np.arange(0.00053,0.0073,0.0002)*(1+self.z)/integrate.quad(cmdist, 0, self.z)[0]*206265
        phot_diff = []
        for radius in radi:
            #print radius
            apertures = CircularAperture(center, r=radius/0.03)
            phot_table_o = aperture_photometry(self.origin, apertures)
            phot_table_d = aperture_photometry(self.deconvolved, apertures)
            phot_table_hst = aperture_photometry(self.hst, apertures)

            tmp = (-2.5*np.log10(phot_table_hst['aperture_sum'][0]/frac_ee(radius/0.03))+zpst) - (-2.5*np.log10(phot_table_o['aperture_sum'][0])+30.0)
            try:
                tmp2 = (-2.5*np.log10(phot_table_hst['aperture_sum'][0]/frac_ee(radius/0.03))+zpst) - (-2.5*np.log10(phot_table_d['aperture_sum'][0])+30.0)
            except:
                tmp2 = np.nan

            # if np.abs(tmp2)/(-2.5*np.log10(phot_table_hst['aperture_sum'][0]/frac_ee(radius/0.03))+zpst))<0.9:
            #     return radius, phot_table_hst['aperture_sum'][0]
            phot_diff.append([ tmp, tmp2])

        return radi, phot_diff

class PDF():
    def __init__(self):
        from matplotlib.backends.backend_pdf import PdfPages

        self.pdf = PdfPages('temp_holder.pdf')

    def plots(self, directory, name,  tmpData, tmpValues):

        def plot(fig, inner_gs, newdata, tmpValues):
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.colorbar import Colorbar
            from matplotlib import collections  as mc

            import scipy.integrate as integrate
            cmap = sns.cubehelix_palette(light=1, as_cmap=True)

            # data = np.array(tmpData)

            rg = fwhm(hst=.1/0.03, conv=0.76/0.03)
            d = 0.0015*(1+0.9375)/integrate.quad(cmdist, 0, 0.9375)[0]*206265

            title  = ['HST', 'Convolved HST', 'FWHM '+ str(tmpValues[3]), 'RMS '+ str(tmpValues[2])]

            vmin=np.min(newdata[1])
            vmax=np.max(newdata[1])

            #plt.subplots(2,3, figsize=(15, 10))
            for i, plot in enumerate(newdata):

                ax = plt.Subplot(fig, inner_gs[i]) #gs[0,i]
                if i == 0:
                    vmin=np.min(newdata[i])
                    vmax=np.max(newdata[i])
                else:
                    vmin=np.min(newdata[1])
                    vmax=np.max(newdata[1])
                # else:
                ax.imshow(plot, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax) #plt.get_cmap('viridis')

                ax.grid(color='r', linestyle='--', linewidth=0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.set_xticks([])
                # ax.set_yticks([])

                ax.text(5,5, title[i], weight='bold', fontsize=5)

                if i == 2:
                    r = fwhm(hst=.1/0.03, conv=tmpValues[3]/0.03)
                    try:
                        ax.text(5,170, '$\lambda$: '+ tmpValues[0].split('/lambd_')[1], weight='bold', fontsize=5)
                    except:
                        pass

                    circle = plt.Circle((15,50), r, fill=False, linewidth=0.75, color='k')
                    lc = mc.LineCollection([[(15-(d/0.06),49.75), (15+d/0.06,49.75)]], linewidths=0.75, colors='k')
                    lc = mc.LineCollection([[(15-(d/0.06),49.75), (15+d/0.06,49.75)]], linewidths=0.75, colors='k')
                    ax.add_artist(circle)
                    ax.add_collection(lc)

                fig.add_subplot(ax)

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        data = np.array(tmpData)

        fig = plt.figure(figsize=(8, 2*data.shape[0]), dpi=400)
        outer_gs = gridspec.GridSpec(data.shape[0], 1)#, height_ratios=[0.05,1,1], width_ratios=[1,1,1])
        outer_gs.update(wspace=0.05, hspace=0.05)

        # data = np.array(tmpData)
        # print data.shape

        for j, newdata in enumerate(data):
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 4,
                    subplot_spec=outer_gs[j], wspace=0.05, hspace=0.05)
            # plt.setp(axes, title=tmpValues[j][0])
            # plt.suptitle(tmpValues[j][0], fontsize=5)
            plot(fig, inner_gs, newdata, tmpValues[j])


        # plt.title(tmpValues[1][0].split('/lambd_')[0], fontsize=5)
        plt.savefig(directory+'/'+name+'.pdf')
        # print directory+'/'+name+'.pdf'

        # plt.savefig(directory.replace('*','')+'/tmp.pdf')
        # plt.savefig(directory.replace('/sfact_4_gres_2', '')+'.pdf')
        plt.close()


    def savePDF(self):
        from tqdm import tqdm

        for fig in tqdm(xrange(1,plt.gcf().number+1)):
            self.pdf.savefig( fig)
        plt.close('all')
        #self.pdf.close()



def check_residual(directory='results/deconv_01.fits', hst_directory='results_2000_2000/', corrections=False, save=None):
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    # adjust wcs for resampled deconvolved data
    #w = pywcs.WCS(deconvolve_h)

    #ra_corr, dec_corr = find_corrections()

    if corrections:
        print ('Correcting for RA and DEC values...')
        deconvolve_h['CRVAL1'] = deconvolve_h['CRVAL1']+5.11235740e-07
        deconvolve_h['CRVAL2'] = deconvolve_h['CRVAL2']-1.24497218e-06
    else:
        pass


def search_cfg(filename, dic):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    for line in lines:
        if len(line) > 2 and '=' in line:
            if line[-1] != '\n':
                line += '\n'
            name, val = line.split("=")
            name, val = name.strip(), val.strip()
            if name in dic:
                return val


def write_cfg(filename, var_dic):
    from copy import deepcopy
    dic = deepcopy(var_dic)#.copy()
    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    out = ''
    for line in lines:
        if len(line) > 2 and '=' in line:
            if line[-1] != '\n':
                line += '\n'
            name, val = line.split("=")
            name, val = string.strip(name), string.strip(val)
            if name in dic:
                if type(dic[name]) == type(''):
                    out += name + " = '" + str(dic[name]) + "'\n"
                else:
                    out += name + ' = ' + str(dic[name]) + '\n'
                del dic[name]
            else:
                out += line
        else:
            out += line
    if len(dic) > 0:
        for name in dic.keys():
            if type(dic[name]) == type(''):
                out += name + " = '" + str(dic[name]) + "'\n"
            else:
                out += name + ' = ' + str(dic[name]) + '\n'

    f = open(filename, "w" )
    f.write(out)
    f.flush()
    f.close()


def makeVBin(directory, data, noise, segmap=None):
    from pymcs.src import _1_get_sky
    import numpy as np
    import os

    # data, deconvolve_h = openFits(directory+'/deconv_01.fits')
    # c, std = _1_get_sky.get_sky_val(data, show=False, range=[0.05,0.95], nbins=35, save=None)
    c = data.shape[0]/2.
    cin = c/2.
    tmp = np.sqrt(data[int(c-cin):int(c+cin),int(c-cin):int(c+cin)]+noise)
    bn = os.path.basename(directory)
    # import matplotlib.pyplot as plt
    # plt.imshow((data/tmp)*self.segmap)
    # plt.colorbar()
    # plt.show()
    if segmap is None:
        indices = np.indices(tmp.shape)
        d= np.array([indices[0].ravel(), indices[1].ravel(), data[int(c-cin):int(c+cin),int(c-cin):int(c+cin)].ravel(), tmp.ravel()]).T
        # print directory.replace(bn, '')+'/vorbin_input.txt'
        with open(directory.replace(bn, '')+'/vorbin_input.txt', 'w+') as datafile_id:
            np.savetxt(datafile_id, d, fmt=['%f','%f', '%f','%f'])
    else:
        idx = np.where(segmap>0)
        # print idx
        indices = np.indices(tmp.shape)
        # print self.segmap.shape
        # print indices[0][idx].ravel().shape
        #[idx] + 0.5
        d= np.array([indices[0][idx].ravel(), indices[1][idx].ravel(), data[idx].ravel(), tmp[idx].ravel()]).T

        with open(directory.replace(bn, '')+'/vorbin_input.txt', 'w+') as datafile_id:
            np.savetxt(datafile_id, d, fmt=['%f','%f', '%f','%f'])

class vorbin():
    '''
    Make vorbin input text file which include the signal and noise of every pixel
    For all the objects in each filters
    '''

    def __return_lambd(self, directory):
        import glob
        lamdbs = glob.glob(directory+'/lambd*')
        return lamdbs[-1]

    def vorbin(self, directory):
        import numpy as np
        import glob
        import os

        for i in glob.glob(directory):
            filedir = self.__return_lambd(i)
            objectID = os.path.basename(i)
            lambdID = os.path.basename(filedir)

            deconvolve, dh = openFits(filedir+'/deconv_01.fits')

            # segdir = '../run/selectedgal/{}/{}/{}/segmap.fits'.format('ultravista-t065-j-sfact_3_gres_2',objectID, 'lambd_000100.0')
            # segmap = pyfits.open(segdir)[0].data
            # self.segmap = segmap.repeat(3, axis = 0).repeat(3, axis = 1).T
            # self.segmap[self.segmap > 1] = 1

            imagedir = search_cfg(filedir+'/config.py', 'FILENAME')[1:-1]
            correlnoise = getnoise(imagedir)
            makeVBin(filedir, deconvolve, correlnoise, None)
