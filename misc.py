import numpy as np
from astropy.io import fits
from astropy.io import ascii
import diagnostics
import glob
from tqdm import tqdm
from matplotlib import gridspec

class fitEllipse:
    from numpy.linalg import eig, inv

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
#         return a

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

        masked = np.zeros(self.segmap.shape)
        masked[self.segmap==0] = np.nan
        if _ellipse(masked, center, axes[0], axes[1], phi) >= _ellipse(masked, center, axes[1], axes[0], phi):
            a, b = axes
        else:
            b, a = axes

        self.axmaj = a
        self.axmin = b
        self.phi = phi
        return center+[a,b,phi]

    def plotEllipse(self, a, b, to):
        t = np.linspace(0, 2*np.pi, 100)
        Ell = np.array([a*np.cos(t) , b*np.sin(t)])
        R_rot = np.array([[np.cos(to) , -np.sin(to)],[np.sin(to) , np.cos(to)]])

        Ell_rot = np.zeros((2,Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

        return Ell_rot

def rainbowb():
    import matplotlib

    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.1, 0.5, 0.5),
                     (0.2, 0.0, 0.0),
                     (0.4, 0.2, 0.2),
                     (0.6, 0.0, 0.0),
                     (0.8, 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
            'green':((0.0, 0.0, 0.0),
                     (0.1, 0.0, 0.0),
                     (0.2, 0.0, 0.0),
                     (0.4, 1.0, 1.0),
                     (0.6, 1.0, 1.0),
                     (0.8, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),
            'blue': ((0.0, 0.0, 0.0),
                     (0.1, 0.5, 0.5),
                     (0.2, 1.0, 1.0),
                     (0.4, 1.0, 1.0),
                     (0.6, 0.0, 0.0),
                     (0.8, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return my_cmap


def _get_lambdir(directory):
    import glob
    lamdbs = glob.glob(directory+'/lambd*')
    return lamdbs[-1]

def _get_segmap(data, savedir=None):
    from pymcs.src.lib import pysex
    from astropy.io import fits
    import matplotlib.pyplot as plt

    dirname = savedir+'/segmap.fits'
    cat = pysex.run(data, params=['X_IMAGE', 'Y_IMAGE'], conf_args={'CHECKIMAGE_TYPE':'SEGMENTATION', \
                                                                    'CHECKIMAGE_NAME': '../'+dirname})

    # open segmap
    segmap = fits.open(dirname)[0].data
    # resize segmap to match deconvolved data
#     plt.imshow(data*segmap.T)
#     plt.show()
    segmap = segmap.repeat(3, axis = 0).repeat(3, axis = 1).T

    segmap[segmap > 1] = 1.
#     segmap[segmap > 1] = 0

    return segmap

def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def createSquareMask(h, w, center=None, width=None):
    mask = np.zeros((h,w))

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]

    mask[center[0]-1:center[0]-1+width, center[1]-1:center[1]-1+width] = 1
    return mask

def returnRGB(directory, offsetdata=False, fact=1, fixoffset=False):

    def _return_offseted_data(offsetsdata, filtname, data):

        bands = ['subaru_IA427', 'subaru_B', 'subaru_IA484', 'subaru_IA505', 'subaru_IA527', 'subaru_V',\
         'subaru_IA624', 'subaru_rp', 'subaru_IA738', 'subaru_zp', 'ultravista_Y', 'ultravista_J', 'ultravista_H']

        if filtname == 'ultravista_Ks':
            dy, dx = [0,0]
        else:
            idx_offset = bands[::-1].index(filtname)
            dy = offsetsdata['dy'].data[idx_offset]
            dx = offsetsdata['dx'].data[idx_offset]

        data = np.roll(data, int(dy*3), 0)
        data = np.roll(data, int(dx*3), 1)
        return data

    from skimage.transform import rescale

    bfilt = glob.glob(directory+'/subaru-V*')[0]
    zfilt = glob.glob(directory+'/subaru-zp*')[0]
    kfilt = glob.glob(directory+'/ultravista-H*')[0]

    def cross_correlate(ndarray):
        from scipy.signal import correlate

        correctedimgs = []
        matchimg = ndarray[0]

        correctedimgs.append(matchimg)
        for i, tmpimg in enumerate(ndarray[1:]):
            correlation = correlate(tmpimg, matchimg, method='fft')
            tmpy, tmpx = np.unravel_index(np.argmax(correlation), correlation.shape)

            dy = correlation.shape[0]/2-tmpy
            dx = correlation.shape[0]/2-tmpx

            tmpimg = np.roll(tmpimg, dy, 0)
            tmpimg = np.roll(tmpimg, dx, 1)

            correctedimgs.append(tmpimg)
            matchimg = tmpimg

        return correctedimgs

    def create_rgb(b, g, r, scalemin, scale='linear'):
        import img_scale
        img = np.zeros((r.shape[0], r.shape[1], 3), dtype=float)

        img[:,:,2] = getattr(img_scale, scale)(b, scale_min=scalemin[0], scale_max=np.max(b))
        img[:,:,1] = getattr(img_scale, scale)(g, scale_min=scalemin[1], scale_max=np.max(g))
        img[:,:,0] = getattr(img_scale, scale)(r, scale_min=scalemin[2], scale_max=np.max(r))

        return img

    b = fits.open(bfilt+'/deconv_01.fits')[0].data
    z = fits.open(zfilt+'/deconv_01.fits')[0].data
    k = fits.open(kfilt+'/deconv_01.fits')[0].data
    if offsetdata:
        b = _return_offseted_data(offsetdata, 'subaru_V', b)
        z = _return_offseted_data(offsetdata, 'subaru_zp', z)
        k = _return_offseted_data(offsetdata, 'ultravista_Ks', k)

    tmpdata = [b, z, k]
    for tmpi, td in enumerate(tmpdata):
        tmpdata[tmpi] = rescale(td, 260/156., mode='reflect')*(156/260.)**2

    hscale = 1./(10**(-(-30+31.4)/-2.5))
    z *= hscale

    if fixoffset:
        k, z, b = cross_correlate(tmpdata[::-1])

    # if type(segmap) is bool:
    #     rgbimg = create_rgb(b, z, k, [np.median(b), np.median(z), np.median(k)])
    # else:
    #     rgbimg = create_rgb(b*segmap, z*segmap, k*segmap, [np.median(b*segmap), np.median(z*segmap), np.median(k*segmap)])

    rgbimg = create_rgb(b, z, k, [np.median(b), np.median(z), np.median(k)])
    return rgbimg

def getHST(directory, medianlist=False):
    import os
    from scipy.signal import convolve2d
    from scipy.signal import correlate
    import matplotlib.pyplot as plt

    def create_rgb(b, g, r, scalemin, scale='linear'):
        import sys
        sys.path.insert(0, './')
        import img_scale
        img = np.zeros((r.shape[0], r.shape[1], 3), dtype=float)

        img[:,:,2] = getattr(img_scale, scale)(b, scale_min=scalemin[0], scale_max=np.max(b))
        img[:,:,1] = getattr(img_scale, scale)(g, scale_min=scalemin[1], scale_max=np.max(g))
        img[:,:,0] = getattr(img_scale, scale)(r, scale_min=scalemin[2], scale_max=np.max(r))

        return img


    filtdirs = [directory+filt for filt in ['/f814w_ps.fits', '/f475w_ps.fits', '/f160w_ps.fits']]

    if all([os.path.isfile(f) for f in filtdirs]):

        kernb = fits.open('/mnt/drivea/run/kernpsf/F475W_kern.fits')[0].data
        kerng = fits.open('/mnt/drivea/run/kernpsf/F606W_kern.fits')[0].data
        kernz = fits.open('/mnt/drivea/run/kernpsf/F814W_kern.fits')[0].data

        hsto = fits.open(directory+'/f475w_ps.fits')[0].data
        hstz = fits.open(directory+'/f814w_ps.fits')[0].data
        hstb = fits.open(directory+'/f606w_ps.fits')[0].data
        hsth = fits.open(directory+'/f160w_ps.fits')[0].data

        zpz = -2.5*np.log10(7.0724e-20)-5*np.log10(8.05975e3) -2.408
        zpb = -2.5*np.log10(1.7855e-19)-5*np.log10(4.74522e3) -2.408
        zpv = -2.5*np.log10(7.8624e-20)-5*np.log10(5.92111e3) -2.408
        zph = -2.5*np.log10(1.9275e-20)-5*np.log10(15369.176) -2.408

        zscale = 1./(10**(-(-30.0+zpz)/-2.5))
        bscale = 1./(10**(-(-30.0+zpb)/-2.5))
        vscale = 1./(10**(-(-30.0+zpv)/-2.5))
        hscale = 1./(10**(-(-30.0+zph)/-2.5))

        hsth *= hscale
        hsto = convolve2d(hsto*bscale, kernb, mode='same')
        hstb = convolve2d(hstb*vscale, kerng, mode='same')
        hstz = convolve2d(hstz*zscale, kernz, mode='same')
        # print (zph,zpv)

        matchimg = hsth
        correctedimgs = []
        for j, tmpimg in enumerate([hstz, hstb]):

            a = np.arange(9).reshape((3,3))
            b = np.arange(9)[::-1].reshape((3,3))

            correlation = correlate(a, b, method='fft')
            # print (type(hstz), type(hstb))
            # plt.subplots(1,2)
            # plt.subplot(121)
            # plt.imshow(tmpimg)
            # plt.subplot(122)
            # plt.imshow(matchimg)
            # plt.show()
            correlation = correlate(tmpimg, matchimg, method='fft')
            tmpy, tmpx = np.unravel_index(np.argmax(correlation), correlation.shape)

            dy = correlation.shape[0]/2-tmpy
            dx = correlation.shape[0]/2-tmpx

            tmpimg = np.roll(tmpimg, int(dy), 0)
            tmpimg = np.roll(tmpimg, int(dx), 1)

            correctedimgs.append(tmpimg)
            matchimg = tmpimg

        hstz = correctedimgs[0]
        hstb = correctedimgs[1]

        if type(medianlist) is type(True):
            imhst = create_rgb(hstb, hstz, hsth, [np.nanmedian(hstb), np.nanmedian(hstz), np.nanmedian(hsth)])
            return imhst
        else:
            imhst = create_rgb(hstb, hstz, hsth, [medianlist[0], medianlist[1], medianlist[2]])
            return [hsto, hstb, hstz, hsth, imhst]
    else:
        return False

def sedfits(directory, binshape):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def _fluxd_to_flux(fluxdensity, efflamdba):
        flux = fluxdensity#np.array([float(i) for i in data])

        flux[::2] *= (2.99e10)/(efflamdba*efflamdba*1e-8)
        flux[1::2] *= (2.99e10)/(efflamdba*efflamdba*1e-8)

        flux *= 10**(-73.6/2.5) # rescale from z=25 to z=-48.6

        return flux

    # y = int(coords[1])
    # x = int(coords[0])
    tmp = ascii.read(directory+'/test_phot/cosmos.cat')
    fluxdensity = np.loadtxt(directory+'/test_phot/cosmos.cat', skiprows = 1)

    filternames = tmp.colnames[1:-1][::2]

    def filtercolors(filternames):
        colors = []
        for fn in filternames:
            if int(fn[1:]) == 72:
                colors.append('bD')
            elif 78 <= int(fn[1:]) <= 83:
                colors.append('m^')
            elif 181 <= int(fn[1:]) <= 199:
                colors.append('gs')
            elif 200 <= int(fn[1:]) <= 259:
                colors.append('ro')
        return colors

    fcolors = filtercolors(filternames)
    # sedid = np.arange(0,36)
    sedid = binshape[:,1]
    _ids = np.arange(np.max(tmp['id'].data)+1)

    ncol = 8
    nrow = 3
    fig = plt.figure(figsize=((ncol+1)*1.4, (nrow+1)*1.4))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1),\
                           bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1))


    efflambda, model = np.loadtxt(directory+'/test_phot/BEST_FITS/cosmos_0.input_res.fit', skiprows=1, unpack=True)
    for i, _id in enumerate(sedid):

        # _id = int(_ids[-(i+1)])
        _id = int(_id)
        wls, bestfit = np.loadtxt(directory+'/test_phot/BEST_FITS/cosmos_{}.fit'.format(_id), skiprows=1, unpack=True)

        inputflux = _fluxd_to_flux(fluxdensity[_id][1:-1], efflambda)

        scalejy = 1e-19
        bestfit *= scalejy#*1e6

        ax = plt.subplot(gs[i])

        ax.plot(wls, bestfit, color='grey')
        ax.axvline(0)
        # ax.scatter(wls, inputflux[::2], color='r', s=2, edgecolors='k')
        for j, fc in enumerate(fcolors):
            ax.errorbar(efflambda[j], inputflux[::2][j], yerr=inputflux[1::2][j], fmt=fc, ecolor ='k', elinewidth=1, capsize=2, mec='k', ms=6)
            ax.scatter(efflambda[j], inputflux[::2][j], color='g', s=2, edgecolors='k')
        ax.set_xlim([3000,40000])
        ax.set_xscale("log")
        # if i not in [0,5,10,15,20]:
        ax.set_yticks([])

    fig.text(0.5, 0.025, 'Wavelength [Angstrom]', ha='center')
    fig.text(0.025, 0.5, 'F$\lambda$ (Arbitrary)', va='center', rotation='vertical')
    plt.show()

def tmpsedfits(directory, binshape):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def _fluxd_to_flux(fluxdensity, efflamdba):
        # efflambda in anstrom

        # with open(filename) as f:
        #     data = f.readlines()[binnumber].split()[1:-1]

    #     efflambda = np.array([4478.34, 4808.39, 6314.68, 5492.89, 9054.35, 16490.2, 12549.7, 10223.6]) # in angstrom

        flux = fluxdensity#np.array([float(i) for i in data])

        flux[::2] *= (2.99e10)/(efflamdba*efflamdba*1e-8)
        flux[1::2] *= (2.99e10)/(efflamdba*efflamdba*1e-8)

        flux *= 10**(-73.6/2.5) # rescale from z=25 to z=-48.6
        # flux *= 1e-32
        # flux[:-3] /= 10**(-6.4/2.5)
        # flux[-3:] /= 10**(-5./2.5)

        return flux

    # y = int(coords[1])
    # x = int(coords[0])
    tmp = ascii.read(directory+'/test_phot/cosmos.cat')
    fluxdensity = np.loadtxt(directory+'/test_phot/cosmos.cat', skiprows = 1)

    filternames = tmp.colnames[1:-1][::2]

    def filtercolors(filternames):
        colors = []
        markers = []
        zorders = []
        for fn in filternames:
            if int(fn[1:]) == 72:
                colors.append('tab:green')
                markers.append('s')
            elif 78 <= int(fn[1:]) <= 83:
                colors.append('green')
                markers.append('s')
                zorders.append(10)
            elif 181 <= int(fn[1:]) <= 199:
                colors.append('tab:purple')
                markers.append('D')
                zorders.append(5)
            elif 256 <= int(fn[1:]) <= 259:
                colors.append('red')
                markers.append('o')
                zorders.append(15)
        return colors, markers, zorders

    fcolors, markers, zorders = filtercolors(filternames)
    # sedid = np.arange(0,36)
    sedid = binshape[:,1]
    _ids = np.arange(np.max(tmp['id'].data)+1)

    ncol = 7
    nrow = 3
    fig = plt.figure(figsize=((ncol+1)*1.4, (nrow+1)*1.4))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1),\
                           bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1))


    efflambda, model, input, unc_obs = np.loadtxt(directory+'/test_phot/best_fits/cosmos_0.0.input_res.fit', skiprows=1, unpack=True)
    for i, _id in enumerate(sedid):

        # _id = int(_ids[-(i+1)])
        _id = int(_id)
        wls, bestfit = np.loadtxt(directory+'/test_phot/best_fits/cosmos_{}.0.fit'.format(_id), skiprows=1, unpack=True)

        inputflux = _fluxd_to_flux(fluxdensity[_id][1:-1], efflambda)

        scalejy = 1e-19
        bestfit *= scalejy#*1e6

        ax = plt.subplot(gs[i])

        ax.plot(wls, bestfit, color='grey', zorder=1)
        ax.axvline(0)
        # ax.scatter(wls, inputflux[::2], color='r', s=2, edgecolors='k')
        for j, (color, marker, zorder) in enumerate(zip(fcolors, markers, zorders)):
            ax.errorbar(efflambda[j], inputflux[::2][j], yerr=inputflux[1::2][j], color=color, marker=marker,\
                        ecolor ='k', elinewidth=1, capsize=2, mec='k', ms=6, zorder=zorder)
            ax.scatter(efflambda[j], inputflux[::2][j], color='g', s=2, edgecolors='k', zorder=zorder)
        ax.set_xlim([3000,40000])
        ax.set_xscale("log")
        # if i not in [0,5,10,15,20]:
        ax.set_yticks([])

    fig.text(0.5, 0.025, 'Wavelength [Angstrom]', ha='center')
    fig.text(0.025, 0.5, 'F$_\lambda$ (Arbitrary)', va='center', rotation='vertical')

    plt.savefig('id119634-sed.jpg'.format(_id), bbox_inches='tight', dpi=300)
    plt.show()

# def sedfits(directory, coords, binmap):
#     import co_added_normalized
#     import matplotlib.pyplot as plt
#
#     y = int(coords[1])
#     x = int(coords[0])
#
#     bestfit = ascii.read(directory+'/test_phot/BEST_FITS/cosmos_{}.fit'.format(int(binmap[y,x])))
#     bestinres = ascii.read(directory+'/test_phot/BEST_FITS/cosmos_{}.input_res.fit'.format(int(binmap[y,x])))
#     wls = bestinres['col1'].data
#
#     inputflux = co_added_normalized._fluxd_to_flux(directory+'/test_phot/cosmos.cat' , int(binmap[y,x])+1, wls)
#
#     scalejy = 1e-19
#     bestfit['col2'] *= scalejy#*1e6
#
#     plt.figure(1)
#     plt.plot(bestfit['col1'], bestfit['col2'], color='darkgrey')
#     plt.scatter(wls, inputflux[::2], color='r', edgecolors='k')
#     plt.errorbar(wls, inputflux[::2], yerr=inputflux[1::2], fmt='o')
#     plt.xlim([3000,80000])
#     plt.xscale("log")
#     plt.xlabel("Angstrom")
#     plt.ylabel("F$\lambda$")
#     plt.show()

def diagplots(segmap, binmap, physvars, photvars):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt

    segmap[segmap==0]=np.nan

    nrow = 2
    ncol = 4

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.rcParams.update({'font.size':5*2})

    fig = plt.figure(figsize=((ncol+1)*1.75, (nrow+1)*1.75))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1),\
           bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1))


    cmap = plt.cm.OrRd_r
    cmap.set_under(color='black')
    ax = plt.subplot(gs[1])
    ax.imshow(segmap, origin='lower', cmap=cmap)

    ax = plt.subplot(gs[2])
    plt.imshow(binmap*segmap, origin='lower', cmap='prism')

    ax = plt.subplot(gs[4])
    ax.imshow(-2.5*np.log10(photvars[1]/photvars[2]), cmap='Spectral_r', origin='lower')
    ax.text(5,5, 'Rest-frame (U-V)')

    ax = plt.subplot(gs[5])
    ax.imshow(photvars[1], cmap='magma_r', origin='lower')
    ax.text(5,5, 'U$_\mathrm{rest}$')

    ax = plt.subplot(gs[6])
    ax.imshow(photvars[2], cmap='magma_r', origin='lower')
    ax.text(5,5, 'V$_\mathrm{rest}$')

    ax = plt.subplot(gs[7])
    ax.imshow(physvars[0], cmap='viridis', origin='lower')
    ax.text(5,5, 'Stellar Mass Density')

    plt.show()
