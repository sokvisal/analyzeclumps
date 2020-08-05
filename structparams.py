
import numpy as np
import misc

'''
Miscellaneous Functions needed to run the co-added normalized analysis
Needed to delete the useless functions and added some comments to others
'''

def createCircularMask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def massmasked(masslimit, mass, array):
    '''
    Return masked array based on the limit of mass.
    Mask is given as nans.
    '''

    m = 10.**mass.ravel()

    newarray = array.ravel().astype('float')
    newarray[m<10.**masslimit] = np.nan
    maskedarray = newarray.reshape(array.shape)

#     print np.argwhere(~np.isnan(newarray)).shape

    return maskedarray

def ellipses(a, b, phi):
    ells = []
    for i in np.arange(1,3):
        t = np.linspace(0, 2*np.pi, 100)
        Ell = np.array([i*a*np.cos(t) , i*b*np.sin(t)])
             #u,v removed to keep the same center location
        R_rot = np.array([[np.cos(phi) , -np.sin(phi)],[np.sin(phi) , np.cos(phi)]])
             #2-D rotation matrix

        Ell_rot = np.zeros((2,Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
        ells.append(Ell_rot)
    return ells

def _ellipse_aperture(masked, center, a, b, phi):
    yi, xi = np.indices(masked.shape)
    yi = yi.ravel()[~np.isnan(masked).ravel()]
    xi = xi.ravel()[~np.isnan(masked).ravel()]

    xc = center[1]
    yc = center[0]

    ell = ((xi-xc)*np.cos(phi)+(yi-yc)*np.sin(phi))**2./a**2 + ((xi-xc)*np.sin(phi)-(yi-yc)*np.cos(phi))**2./b**2

    tmpidx = np.where(ell<1)[0]
    lm = masked.ravel()

    sely = yi[tmpidx]
    selx = xi[tmpidx]
    selm = 10**(lm[~np.isnan(masked).ravel()][tmpidx])

    yc = np.sum(sely*selm)/np.sum(selm)
    xc = np.sum(selx*selm)/np.sum(selm)


    y2 = np.sum(selm*sely**2.)/np.sum(selm) - yc**2
    x2 = np.sum(selm*selx**2.)/np.sum(selm) - xc**2
    xy = np.sum(selm*selx*sely)/np.sum(selm) - xc*yc
    pa = np.arctan2(2*xy, x2-y2)/2.
    # theta0 = np.rad2deg(pa)
    return yc, xc, pa# len(tmpidx)

def _return_cm(mass, center, a, b, phi):
    ############## Central Mass ######################
    # note center is the center of segmap
    c = mass.shape[0]/2

    yc, xc, pa = _ellipse_aperture(mass, center , a/2., b/2., phi)

    # yc = np.sum(yi*msel)/np.sum(msel)
    # xc = np.sum(xi*msel)/np.sum(msel)
    #
    # y2 = np.sum(msel*yi**2.)/np.sum(msel) - yc**2
    # x2 = np.sum(msel*xi**2.)/np.sum(msel) - xc**2
    # xy = np.sum(msel*xi*yi)/np.sum(msel) - xc*yc
    # pa = np.arctan2(2*xy, x2-y2)/2.
    # theta0 = np.rad2deg(pa)
    #
    # a = np.sqrt((x2+y2)/2. + np.sqrt(((x2-y2)/2.)**2.+xy**2))
    # b = np.sqrt((x2+y2)/2. - np.sqrt(((x2-y2)/2.)**2.+xy**2))

    y, x = np.indices(mass.shape)
    m = 10**mass.ravel()

    yi = y.ravel()[~np.isnan(m)]
    xi = x.ravel()[~np.isnan(m)]

    # import matplotlib.pyplot as plt
    # plt.scatter(xi,yi)
    # plt.ylim([0,156])
    # plt.xlim([0,156])
    # plt.gca().set_aspect(1./1)
    # plt.show()
    return yi, xi, yc, xc, pa#, theta0, a, b

def setup_profile(center, a, b, phi, physvars, photvars, zphot, weightmap=False):
    '''
    This function create the co-added normalized profile. First the function finds the COM and distribution
    of mass in the galaxy. This is used to find the half-light/mass radius of the galaxy. The average value
    within the half-light/mass radius of the galaxy is used to normalized the pixel's values (the distance
    of the pixel the COM is also normalized by the halft-light/mass radius)

    prop: the variable containing the physical properties of the galaxy (stellar mass, age, sfr)
    '''

    lm = physvars[0]
    lsfr = physvars[1]
    agew = physvars[2]

    uvrest = -2.5*np.log10(photvars[1]/photvars[2])

    yi, xi, yc, xc, pa = _return_cm(lm, center, a, b, phi)
    # yc = 76.87529007968351
    # xc = 79.06927075639413
    # print yc, xc
    cmask = createCircularMask(lm.shape[0], lm.shape[0], radius=21)
    masstot = np.log10(np.nansum(10**(lm*cmask)))
    yc, xc = center
    # phi = pa
    e = np.sqrt(1-b**2/a**2)

    def halflightR(data, mass=False):
        '''
        Return normalized array wrt half-light/half-mass radius
        Set mass to True if working with half-mass radius
        '''
        summ = []
        npix = []

        # print data.shape, len(xi)
        if mass:
            arange =  np.arange(1,4.*a)
        else:
            arange =  np.arange(1,.8*a)
        for i in arange:
            b = i*np.sqrt(1-e**2)
            ell = ((xi-xc)*np.cos(phi)+(yi-yc)*np.sin(phi))**2./i**2 + ((xi-xc)*np.sin(phi)-(yi-yc)*np.cos(phi))**2./(b)**2.
            tmpidx = np.where(ell<1)[0]

            summ.append( np.sum(data[tmpidx]))
            npix.append(len(tmpidx))

        maxidx = np.argmax(summ)
        hidx = int(np.argmin(abs(summ[:maxidx]-summ[maxidx]/2.)))
        qre = arange[hidx]
        qnorm = summ[hidx]/npix[hidx]

        # import matplotlib.pyplot as plt
        # plt.scatter(arange, summ)
        # plt.axvline(qre)
        # plt.show()

        # anumean = (summ[2*hidx-1]-summ[hidx])/(npix[2*hidx-1]-npix[hidx])
        # print np.log10(anumean/qnorm)

        return qnorm, qre, None

    def _re_cutoff(array, qre):
        # e = b/a

        b = qre*np.sqrt(1-e**2)
        ell = ((xi-xc)*np.cos(phi)+(yi-yc)*np.sin(phi))**2./(qre)**2 \
                + ((xi-xc)*np.sin(phi)-(yi-yc)*np.cos(phi))**2./(b)**2.
        tmpidx = np.where(ell<1)[0]
        return tmpidx

    def norm_sfrd(lsfr, re):
        smask = 10**(lsfr[~np.isnan(lsfr)].ravel())
        idx = _re_cutoff(smask, re)

        normsfr = smask/np.nanmedian(smask[idx])
        return np.log10(normsfr)

    def angular_distance(zp):
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        angdist =  cosmo.angular_diameter_distance(zp).value*1000./206265 # kpc per arcsec
        return angdist

    def cd(lm, re):
        mmask = 10**(lm[~np.isnan(lm)].ravel())

        idx = _re_cutoff(mmask, re)
        totalmass = np.sum(mmask[idx])
        angdist = angular_distance(zphot)
        centdens = totalmass/(len(idx)*angdist*0.05**2)
        return np.log10(centdens)

    def ssfr_ratio(m, lsfr, re):
        mmask = 10**(m[~np.isnan(m)].ravel())
        smask = 10**(lsfr[~np.isnan(lsfr)].ravel())

        idx = _re_cutoff(smask, re)
        tmpidx = _re_cutoff(smask, 2.5*re)
        newidx = np.array([x for x in tmpidx if not x in idx])
        # print idx, newidx

        ssfr_i = np.sum(smask[idx])/np.sum(mmask[idx])
        ssfr_o = np.sum(smask[newidx])/np.sum(mmask[newidx])

        return ssfr_i, ssfr_o


    def mapNorm(ndarray, masked_coord):
        ''' ndarray: yi, xi, norm_r, norm_quan, uvmap '''
        if masked_coord is None:
            return [list(d) for d in ndarray]
        else:
            return [list(d[masked_coord]) for d in ndarray]

    def _recutoff(array, radius):
        return np.where(array<np.log10(radius))[0]


    r = np.sqrt((yi-yc)**2+(xi-xc)**2)
    import matplotlib.pyplot as plt
    gal_par = [xc, yc]
    norm_par = []
    outerflux = []
    for i, physvar in enumerate(physvars[:1]):
        dmask = 10**(physvar[~np.isnan(physvar)].ravel())
        norm, re, anunorm = halflightR(dmask,  mass=True)

        if not isinstance(weightmap, bool):
            dnorm = np.log10(dmask*weightmap[0][~np.isnan(physvar)].ravel()/norm)
        else:
            dnorm = np.log10(dmask/norm)
        rnorm = np.log10(r/re)

        re2idx = _re_cutoff(dnorm, 2.5*re)

        # plt.scatter(rnorm[re2idx], dnorm[re2idx])
        # plt.scatter(rnorm[newidx], dnorm[newidx])
        # plt.xlim([-2.5,1])
        # plt.ylim([-2.5,1])
        # plt.axvline(0)
        # plt.gca().set_aspect(1/1.)
        # plt.show()

        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-lm[~np.isnan(lm)].ravel())
        ell_h, ell_f  = ellipses(re, re*np.sqrt(1-e**2), phi)
        tmp = mapNorm([yi, xi, rnorm, dnorm, uvrest[~np.isnan(lm)].ravel(), physvar[~np.isnan(physvar)].ravel(), sfrnorm, agew[~np.isnan(agew)].ravel()], re2idx)

        gal_par.insert(len(gal_par), ell_h)
        gal_par.insert(len(gal_par), ell_f)
        norm_par.append(tmp) #[list(dnorm), list(rnorm)])
        outerflux.append(anunorm)

    import matplotlib.pyplot as plt
    for i, photvar in enumerate(photvars):
        dmask = photvar[~np.isnan(photvar)].ravel()
        norm, re, anunorm = halflightR(dmask)

        if not isinstance(weightmap, bool):
            dnorm = np.log10(dmask*weightmap[i+1][~np.isnan(photvar)].ravel()/norm)
        else:
            dnorm = np.log10(dmask/norm)
        rnorm = np.log10(r/re)
        re2idx = _re_cutoff(dnorm,  2.5*re)

        if not i:
            angdist = angular_distance(zphot)
            rkpc = (1./angdist)/0.05
            ssfr_i, ssfr_o = ssfr_ratio(lm, lsfr, rkpc)
            mcd_re = cd(lm, rkpc)


        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-lm[~np.isnan(lm)].ravel())
        ell_h, ell_f  = ellipses(re, re*np.sqrt(1-e**2), phi)
        tmp = mapNorm([yi, xi, rnorm, dnorm, uvrest[~np.isnan(lm)].ravel(), photvar[~np.isnan(photvar)].ravel(), sfrnorm, agew[~np.isnan(agew)].ravel()], re2idx)

        gal_par.insert(len(gal_par), ell_h)
        gal_par.insert(len(gal_par), ell_f)
        norm_par.append(tmp)
        outerflux.append(anunorm)

    return masstot, gal_par, norm_par, [mcd_re, ssfr_i, ssfr_o], outerflux

########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################


def setup_profile_old(prop, phot_vars, zphot, phi):
    '''
    This function create the co-added normalized profile. First the function finds the COM and distribution
    of mass in the galaxy. This is used to find the half-light/mass radius of the galaxy. The average value
    within the half-light/mass radius of the galaxy is used to normalized the pixel's values (the distance
    of the pixel the COM is also normalized by the halft-light/mass radius)

    prop: the variable containing the physical properties of the galaxy (stellar mass, age, sfr)
    '''

    stellar_mass = prop[0]
    lsfr = prop[1]
    agew = prop[2]

    # m = 10.**stellar_mass.ravel()
    # masslim = 2.
    # idx = np.where(m>10.**masslim)

    uvrest = -2.5*np.log10(phot_vars[1]/phot_vars[2])#massmasked(masslim, stellar_mass, -2.5*np.log10(uflux/vflux))

    # umag = uflux
    # vmag = vflux
    m = stellar_mass#massmasked(masslim, stellar_mass, stellar_mass)

    def _return_cm(mass):
        ############## Central Mass ######################
        c = mass.shape[0]/2
        cint = c/4

        y, x = np.indices(mass.shape)
        y = y[c-cint:c+cint, c-cint:c+cint]
        x = x[c-cint:c+cint, c-cint:c+cint]

        m = 10**mass[c-cint:c+cint, c-cint:c+cint].ravel()
        mcentral = 10**mass.ravel()
        idx = np.argwhere(~np.isnan(m))

        yi = y.ravel()[idx]
        xi = x.ravel()[idx]
        msel = m[idx]

        yc = np.sum(yi*msel)/np.sum(msel)
        xc = np.sum(xi*msel)/np.sum(msel)

        y2 = np.sum(msel*yi**2.)/np.sum(msel) - yc**2
        x2 = np.sum(msel*xi**2.)/np.sum(msel) - xc**2
        xy = np.sum(msel*xi*yi)/np.sum(msel) - xc*yc
        pa = np.arctan2(2*xy, x2-y2)/2.
        theta0 = np.rad2deg(pa)

        a = np.sqrt((x2+y2)/2. + np.sqrt(((x2-y2)/2.)**2.+xy**2))
        b = np.sqrt((x2+y2)/2. - np.sqrt(((x2-y2)/2.)**2.+xy**2))

        y, x = np.indices(mass.shape)
        m = 10**mass.ravel()
        idx = np.argwhere(~np.isnan(m))

        yi = y.ravel()[idx]
        xi = x.ravel()[idx]

        return yi, xi, yc, xc, theta0, a, b
    yi, xi, yc, xc, pa, mwa, mwb = _return_cm(m)

    massmask = misc.createCircularMask(len(phot_vars[0]), len(phot_vars[0]), [xc, yc], 21)
    masstot = np.log10(np.nansum(10**(stellar_mass*massmask)))

    # import matplotlib.pyplot as plt
    # plt.imshow(massmask*stellar_mass)
    # plt.show()
    # TODO: need to make sure a/b is the mode and the extent of the distribution
    def galparams(stellar_mass, diagnostic=False):
        import matplotlib.pyplot as plt

        r = np.sqrt((yi-yc)**2 + (xi-xc)**2)
        theta = np.rad2deg(np.arctan2((yi-yc), (xi-xc)))
        theta[theta<-90] += 180.
        theta[theta>90] -= 180.
        # if len(theta[theta>0]) > len(theta[theta<0]):
        #     sign = 1.
        # else:
        #     sign = -1.
        # theta[theta<0] += 90.

        tbinsize = 5
        bintheta = np.arange(-90,90+tbinsize,tbinsize)
        rbinsize = 2
        binr = np.arange(0,50+rbinsize,rbinsize)

        htheta = np.histogram(theta, bins = bintheta, density=True)
        hr = np.histogram(r, bins = binr, density=True)

        from scipy import interpolate
        from sklearn import mixture
        rs = np.arange(rbinsize/2., 50+rbinsize/2., rbinsize)
        ts = np.arange(-90+tbinsize/2., 90+tbinsize/2., tbinsize)

        samples = np.dstack((rs,hr[0]))[0]

        def getbimodal(data, xbins, angle=False):
            from scipy import interpolate
            from sklearn import mixture

            def gauss_function(x, amp, x0, sigma):
                return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

            if angle:
                gmix = mixture.GaussianMixture(n_components = 2)
                fitted = gmix.fit(data) # data is shaped as (len, 1)

                 # Construct function manually as sum of gaussians
                gmm_sum = np.full_like(xbins, fill_value=0, dtype=np.float32)
                for m, c, w in zip(fitted.means_.ravel(), fitted.covariances_.ravel(), fitted.weights_.ravel()):
                    gauss = gauss_function(x=xbins, amp=1, x0=m, sigma=np.sqrt(c))
                    gmm_sum += gauss / np.trapz(gauss, xbins) * w

                    # if angle:
                    minidx = np.argmin(fitted.weights_)
                    mindis = fitted.means_[minidx][0]
                    maxidx = np.argmax(fitted.weights_)
                    maxdis = fitted.means_[maxidx][0]

                return np.max([maxdis, mindis]), gmm_sum
            else:
                gmix = mixture.GaussianMixture(n_components = 1)
                fitted = gmix.fit(data) # data is shaped as (len, 1)

                 # Construct function manually as sum of gaussians
                gmm_sum = np.full_like(xbins, fill_value=0, dtype=np.float32)
                for m, c, w in zip(fitted.means_.ravel(), fitted.covariances_.ravel(), fitted.weights_.ravel()):
                    gauss = gauss_function(x=xbins, amp=1, x0=m, sigma=np.sqrt(c))
                    gmm_sum += gauss / np.trapz(gauss, xbins) * w

                    modedist = fitted.means_[0]
                return modedist[0], max(data)[0], gmm_sum

        b, a, gmm_r_sum = getbimodal(r, rs)
        to, gmm_t_sum = getbimodal(theta, ts, angle=True)

        # print b, a, to

        if diagnostic:
            plt.subplots(1,2, figsize=(6,3))
            plt.subplot(1,2,1)
            plt.hist(theta, bins = bintheta, density=True)
            plt.title('Angle distribution')
    #         plt.plot(bintheta, g_fit(bintheta))
            plt.plot(ts, gmm_t_sum, color="crimson", lw=4, label="GMM")

            plt.subplot(1,2,2)
            plt.hist(r, bins = binr, density=True)
            plt.plot(rs, gmm_r_sum, color="crimson", lw=4, label="GMM")
            plt.title('R distribution')
            plt.show()

#         theta = g_fit.mean.value
        return to, a, b

    def ellipses(a, b, to):
        # to = np.deg2rad(to)

        ells = []
        for i in np.arange(1,3):
            t = np.linspace(0, 2*np.pi, 100)
            Ell = np.array([i*a*np.cos(t) , i*b*np.sin(t)])
                 #u,v removed to keep the same center location
            R_rot = np.array([[np.cos(to) , -np.sin(to)],[np.sin(to) , np.cos(to)]])
                 #2-D rotation matrix

            Ell_rot = np.zeros((2,Ell.shape[1]))
            for i in range(Ell.shape[1]):
                Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            ells.append(Ell_rot)
        return ells

    from photutils import EllipticalAperture
    from photutils import aperture_photometry
    import copy

    sm_unmask = copy.deepcopy(stellar_mass)

    #h = stellar_mass.shape[0]
    #w = stellar_mass.shape[0]

    to, a, b = galparams(stellar_mass)
    to = pa
    # e = b/a
    e = np.sqrt(1-mwb**2/mwa**2)
    # print 'gal params: ', to, b, a, mwb, mwa

    def halflightR(data, mass=False):
        '''
        Return normalized array wrt half-light/half-mass radius
        Set mass to True if working with half-mass radius
        '''
        summ = []
        npix = []

        if mass:
            maxr = a*2
        else:
            maxr = int(a*2)
        # print data.shape, len(xi)
        for i in np.arange(1,maxr):
            b = i*np.sqrt(1-e**2)
            ell = ((xi-xc)*np.cos(to)+(yi-yc)*np.sin(to))**2./i**2 + ((xi-xc)*np.sin(to)-(yi-yc)*np.cos(to))**2./b**2.
            tmpidx = np.where(ell<1)[0]

            summ.append(sum(data[tmpidx]))
            npix.append(len(tmpidx))

        maxidx = np.argmax(summ)
        hidx = np.argmin(abs(summ[:maxidx]-summ[maxidx]/2.))+1
        qre = np.arange(1,maxr)[hidx]
        qnorm = summ[hidx]/npix[hidx]

        return qnorm, qre

    def _re_cutoff(array, qre):
        b = qre*np.sqrt(1-e**2)
        ell = ((xi-xc)*np.cos(to)+(yi-yc)*np.sin(to))**2./(qre)**2 \
                + ((xi-xc)*np.sin(to)-(yi-yc)*np.cos(to))**2./(b)**2.
        tmpidx = np.where(ell<1)[0]
        return tmpidx

    # mravel = m.ravel()
    # mmask = 10**(m[~np.isnan(m)].ravel())
    # mnorm, mre = halflightR(mmask, mass=True)

    r = np.sqrt((yi-yc)**2+(xi-xc)**2).ravel()

    def mapNorm(ndarray, masked_coord):
        ''' ndarray: yi, xi, norm_r, norm_quan, uvmap '''
        if masked_coord is None:
            return [list(d) for d in ndarray]
        else:
            return [list(d[masked_coord]) for d in ndarray]

    def norm_sfrd(lsfr, re):
        smask = 10**(lsfr[~np.isnan(lsfr)].ravel())
        idx = _re_cutoff(smask, re)

        normsfr = smask/np.nanmedian(smask[idx])
        return np.log10(normsfr)

    def angular_distance(zp):
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        angdist =  cosmo.angular_diameter_distance(zp).value*1000./206265 # kpc per arcsec
        return angdist

    def cd(lm, re):
        mmask = 10**(lm[~np.isnan(lm)].ravel())

        idx = _re_cutoff(mmask, re)
        totalmass = np.sum(mmask[idx])
        angdist = angular_distance(zphot)
        centdens = totalmass/(len(idx)*angdist*0.05**2)
        return np.log10(centdens)

    def ssfr_ratio(m, lsfr, re):
        mmask = 10**(m[~np.isnan(m)].ravel())
        smask = 10**(lsfr[~np.isnan(lsfr)].ravel())

        idx = _re_cutoff(smask, re)
        tmpidx = _re_cutoff(smask, 2*re)
        newidx = np.array([x for x in tmpidx if not x in idx])
        # print idx, newidx

        ssfr_i = np.sum(smask[idx])/np.sum(mmask[idx])
        ssfr_o = np.sum(smask[newidx])/np.sum(mmask[newidx])

        return ssfr_i, ssfr_o

    gal_par = [xc, yc]
    norm_par = []
    res = []
    for i, physvar in enumerate(prop[:1]):
        dmask = 10**(physvar[~np.isnan(physvar)].ravel())
        norm, re = halflightR(dmask, mass=True)
        res.append(re)

        dnorm = np.log10(dmask/norm)
        rnorm = np.log10(r/re)
        re2idx = _re_cutoff(dnorm, 2.5*re)

        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-m[~np.isnan(m)].ravel())
        ell_h, ell_f  = ellipses(re, re*np.sqrt(1-e**2), to)
        tmp = mapNorm([yi, xi, rnorm, dnorm, uvrest[~np.isnan(m)].ravel(), physvar[~np.isnan(physvar)].ravel(), sfrnorm, agew[~np.isnan(agew)].ravel()], re2idx)

        gal_par.insert(len(gal_par), ell_h)
        gal_par.insert(len(gal_par), ell_f)
        norm_par.append(tmp) #[list(dnorm), list(rnorm)])


    import matplotlib.pyplot as plt
    for i, photvar in enumerate(phot_vars):
        dmask = photvar[~np.isnan(photvar)].ravel()

        norm, re = halflightR(dmask)
        res.append(re)

        dnorm = np.log10(dmask/norm)
        rnorm = np.log10(r/re)
        re2idx = _re_cutoff(dnorm,  2.5*re)

        if not i:
            angdist = angular_distance(zphot)
            rkpc = (1./angdist)/0.05
            ssfr_i, ssfr_o = ssfr_ratio(m, lsfr, rkpc)
            mcd_re = cd(m, rkpc)

        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-m[~np.isnan(m)].ravel())
        ell_h, ell_f  = ellipses(re, re*np.sqrt(1-e**2), to)
        tmp = mapNorm([yi, xi, rnorm, dnorm, uvrest[~np.isnan(m)].ravel(), photvar[~np.isnan(photvar)].ravel(), sfrnorm, agew[~np.isnan(agew)].ravel()], re2idx)

        gal_par.insert(len(gal_par), ell_h)
        gal_par.insert(len(gal_par), ell_f)
        norm_par.append(tmp)

    return masstot, gal_par, norm_par, [mcd_re, ssfr_i, ssfr_o], res
