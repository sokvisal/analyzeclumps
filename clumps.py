import numpy as np
from astropy.io import ascii

import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os

import sys
sys.path.insert(0, '/hpcstorage/sok/run/cosmos/analyzeclumps')

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

from operator import itemgetter
from astropy.io import fits


import structparams
import normprofiles
import old_normporfiles
import misc


sfgs_cat = ascii.read('/hpcstorage/sok/run/cosmos_sfgs.dat') #cosmos_final_sfgcat
fout = ascii.read('/hpcstorage/sok/run/UVISTA_final_v4.1_selected.fout')
_ids = sfgs_cat['id'].data

def getMSFR(idnum):
    _ids = sfgs_cat['id'].data
    idx = np.where(_ids==idnum)[0]

    idsfout = fout['col1'].data
    idxfout = np.where(idsfout==idnum)[0]

    idlfast_mass = sfgs_cat['lm'].data[idx]
    cplfast_mass = fout['col7'].data[idxfout]

    lsfr = sfgs_cat['lsfr'].data[idx]
    hmag = sfgs_cat['H'].data[idx]
    bmag = sfgs_cat['B'].data[idx]
    zmag = sfgs_cat['zp'].data[idx]

    umv = sfgs_cat['umv'].data[idx]

    return idlfast_mass[0], cplfast_mass[0], lsfr[0], hmag[0], bmag[0], zmag[0], umv[0]

def sfrw_age(age, sfr):
    from scipy.integrate import quad

    a = 10**(age)
    s = 10**(sfr)

    def age_sfr(x, a, s):
        return s*(a-x)

    def total_sfr(x, s):
        return s

    if sfr == -99.0:
        return sfr
    else:
        bulkage = quad(age_sfr, 0, a, args=(a,s))[0]
        tsfr = quad(total_sfr, 0, a, args=(s))[0]

        return np.log10(bulkage/tsfr)

def _get_ellipse_params(segmapdir):
    from astropy.io import fits
    hdr = fits.open(segmapdir)[0].header

    return hdr['XC'], hdr['YC'], hdr['SEMIMAJ'], hdr['SEMIMIN'], hdr['PA']

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

    y, x, surest, noise = np.loadtxt(directory+'/{}/vorbin_input.txt'.format(urest)).T
    y, x, svrest, noise = np.loadtxt(directory+'/{}/vorbin_input.txt'.format(vrest)).T
#     surest = surest.clip(0.01)
#     svrest = svrest.clip(0.01)
    return surest, svrest

def stellarPopMaps(directory, path):

    tile = directory.split('/')[-2][1:]
    idnum = int(os.path.basename(directory).split('_')[1].split('-')[1])
    zp = float(os.path.basename(directory).split('_')[2].split('-')[1])

    surest, svrest = _return_photometry(directory, zp)
    y, x, sNIR, noise = np.loadtxt(directory+'/ultravista-H/vorbin_input.txt').T

    binNum = np.loadtxt(directory+'/vorbin_output.txt')
    size = 156

    binids, la, lm, lsfr, umag, vmag, l2800, chi2 = np.loadtxt(directory+'/test_phot/cosmos.fout', usecols=(0,4,6,7,10,11,12,13), unpack=True)
    hflux = np.loadtxt(directory+'/test_phot/cosmos.cat', usecols=(23), unpack=True)
    normlm = (lm-min(lm))/(max(lm)-min(lm))
    normH = (hflux-min(hflux))/(max(hflux)-min(hflux))
    normH = normH.clip(0.05)

    weighted_chi2 = []
    pixcounts= 0
    for i, binid in enumerate(binids):
        idx = np.argwhere(binNum==binid)
        npix = idx.shape[0]
        pixcounts += npix
        weighted_chi2.append(chi2[i]*npix)
    weighted_chi2 = np.array(weighted_chi2)

    if np.nanmean(chi2)>5:
        return False, False
    else:
        lsfr[lsfr==-99.0] = -3.
        l2800 = 10**(((5*np.log10(cosmo.luminosity_distance(zp).value*1e5)+l2800)-25.)/(-2.5))
        umag = 10**(((5*np.log10(cosmo.luminosity_distance(zp).value*1e5)+umag)-25.)/(-2.5))
        vmag = 10**(((5*np.log10(cosmo.luminosity_distance(zp).value*1e5)+vmag)-25.)/(-2.5))

        # umag,vmag = np.loadtxt(directory+'/test_phot/OUTPUT/cosmos.153-155.rf', usecols=(5,6), unpack=True)
        # l2800 = np.loadtxt(directory+'/test_phot/OUTPUT/cosmos.219-153.rf', usecols=(5,), unpack=True)
        # l2800[l2800==-99.] = 1e-10
        # umag[umag==-99.] = 1e-10
        # vmag[vmag==-99.] = 1e-10

        binmap = np.ones((size,size))
        tmpvars = np.zeros((1,size,size))*np.nan
        physvars = np.zeros((3,size,size))*np.nan
        photvars = np.zeros((3,size,size))*np.nan

        maxflux = np.max(sNIR)
        minflux = np.min(sNIR)

        binshape = []
        for i, binid in enumerate(binids):
            idx = np.argwhere(binNum==binid)
            npix = idx.shape[0]
            binshape.append([npix, binid])

            coords = zip(y[idx].astype(int),x[idx].astype(int))
            ny, nx = zip(*coords)

            binmap[ny, nx] = binid
            normu = 1+(surest-np.mean(surest))/(np.max(surest)-np.min(surest))
            normv = 1+(svrest-np.mean(svrest))/(np.max(svrest)-np.min(svrest))
            normNIR = 1+(sNIR-np.mean(sNIR))/(np.max(sNIR)-np.min(sNIR))

            znormNIR = (sNIR-min(sNIR))/(max(sNIR)-min(sNIR))
            znormUV = (surest-min(surest))/(max(surest)-min(surest))
            znormV = (svrest-min(svrest))/(max(svrest)-min(svrest))

            if not np.any(np.isnan([lm[i], l2800[i], umag[i], vmag[i]])):
                ubinscale = 1+(np.mean(surest[idx])-np.mean(surest))/(np.max(surest)-np.min(surest))#surest[idx]/np.sum(surest[idx])#1+(surest[idx]-np.mean(surest[idx]))/(np.max(surest)-np.min(surest))
                vbinscale = 1+(np.mean(svrest[idx])-np.mean(svrest))/(np.max(svrest)-np.min(svrest))#svrest[idx]/np.sum(svrest[idx])#1+(svrest[idx]-np.mean(svrest[idx]))/(np.max(svrest)-np.min(svrest))
                nirscale = 1+(np.mean(sNIR[idx])-np.mean(sNIR))/(np.max(sNIR)-np.min(sNIR))

                normIR =  znormNIR[idx].clip(0.025)/1.
                normUV =  znormUV[idx].clip(0.025)/1.
                normV =  znormV[idx].clip(0.025)/1.
                Hscale = normH[i]/1.

                scalemass = np.sum(10**lm[i]*normIR)/np.sum(10**lm[i]*npix)
                physvars[0, ny, nx] = lm[i]#np.log10(10**lm[i]*normIR/scalemass)
                physvars[1, ny, nx] = lsfr[i]
                physvars[2, ny, nx] = 1./scalemass #np.log10(10**lm[i]*nirscale/scalemass) #sfrw_age(la[i], lsfr[i])#-np.log10(npix)

                tmpvars[0, ny, nx] = sNIR[idx]

                photvars[0, ny,nx] = l2800[i]
                ufact = np.sum(umag[i]*normUV)/np.sum(umag[i]*npix)
                photvars[1, ny,nx] = umag[i]#*normUV/ufact#*(normu[idx]/ubinscale)#/npix *normUV/ufact#
                vfact = np.sum(vmag[i]*normV)/np.sum(vmag[i]*npix)
                photvars[2, ny,nx] = vmag[i]#*normV/vfact#*(normv[idx]/vbinscale)#/npix *normV/vfact#

        tmpvars[0][np.isnan(tmpvars[0])] = 0.
        tmpy, tmpx = np.unravel_index(np.argmax(tmpvars[0]), tmpvars[0].shape)
        return [physvars, photvars, binmap, binshape], [tmpy, tmpx]

def retrieved_maps(directories, path):

    master_umap = []
    master_vmap = []
    master_mmap = []

    test_lst = []
    physicals = []
    selzp = []

    newcat_clumps = []

    tile = directories.split('/')[-2][1:]
    badids = []#np.loadtxt('../run/images/{}/ignore.list'.format(tile))

    badcounts = 0
    tmpdirs = [idnames for idnames in glob.glob(directories)[:] if len(glob.glob(idnames+'/*-*'))==14]
#     print [int(os.path.basename(idnames.split('_')[1].split('-')[1])) for idnames in glob.glob(directories)[:] if len(glob.glob(idnames+'/*-*'))==14]
    for d in tmpdirs[:200]: #glob.glob(directories)[:]
        idnum = int(os.path.basename(d).split('_')[1].split('-')[1])
        zp = float(os.path.basename(d).split('_')[2].split('-')[1])
        tmpmass = getMSFR(idnum)[0]

        if os.path.isfile(d+'/test_phot/cosmos.fout'):# tmpmass<10. and zp>1.:
            tmpxc, tmpyc, a, b, phi = _get_ellipse_params('./{}/a{}/watershed_segmaps/_id-{}.fits'.format(path, tile, idnum))
            segmap = fits.open('./{}/a{}/watershed_segmaps/_id-{}.fits'.format(path, tile, idnum))[0].data
            segmap[segmap==0] = np.nan
            offsets = ascii.read('./{}/a{}/offsets/_id-{}.dat'.format(path, tile, idnum) )
            decoffsets = ascii.read('./{}/a{}/offsets/_id-{}-dec.dat'.format(path, tile, idnum) )
            dys = offsets['dy'].data*3+decoffsets['dy'].data
            dxs = offsets['dx'].data*3+decoffsets['dx'].data
            rgbimg = misc.returnRGB(d, offsetdata=[dys, dxs], fixoffset=False)

            boolcheck, coords = stellarPopMaps(d, path)
            if type(boolcheck) == type(True):
                badcounts += 1
            else:
                physvars, photvars, binmap, binshape = boolcheck
                binshape = np.array(sorted(binshape, key=itemgetter(1))[:21])

                physvars *= segmap
                photvars *= segmap
                mass, params, normmaps, diagnostics, outerflux =\
                                structparams.setup_profile([tmpyc, tmpxc], a, b, phi, physvars, photvars, zp)
                # mass, params, normmaps, diagnostics, res = old_normporfiles.coadd_profile(physvars, photvars, zp)


                maps = [physvars[0], physvars[1], photvars[1], photvars[2], -2.5*np.log10(photvars[1]/photvars[2]), binmap]
                galmass_idl, galmass_c, galsfr, hmag, bmag, zmag, umv = getMSFR(idnum)

                selzp.append(zp)
                physicals.append([galmass_idl, galmass_c, mass, galsfr, np.log10(np.nansum(10**physvars[1]))])

                # # 'radial_profiles'
                # # plotting the co-added normalized profile
                fig, clumpids, ccs, agew_rnorm = normprofiles.make_profile(rgbimg, maps, normmaps,\
                                params, [zp, idnum], tile, outerflux, res=[None, None, None, None],\
                                                                           savedir=False, showplot=False)
                # fig, clumpids, clumpiness, ccs = old_normporfiles.caddnorm_plot(rgbimg, maps, normmaps, params, idnum, res)


                newcat_clumps.append([idnum, zp,  galmass_idl, mass, galsfr] + clumpids + ccs + diagnostics)

#                 if abs(galmass_idl-mass)>0.3:
                print (d, galmass_idl, mass)
#                 misc.tmpsedfits(d, binshape)

                master_mmap.append(normmaps[0][2:])
                master_umap.append(normmaps[1][2:])
                master_vmap.append(normmaps[2][2:])

    return [master_umap, master_vmap, master_mmap], physicals, selzp, test_lst, newcat_clumps



def createCat(decpath, tile):
    dirname = './{}/a{}/_id*'.format(decpath, tile)
    normalizedprofile, physicals, selzp, test_lst, ids_clumps = retrieved_maps(dirname, decpath)

    ids_clumps = np.array(ids_clumps)

    from astropy.table import Table
    ids_clumps_cat = Table([ids_clumps[:,i] for i in range(18)],\
                           names=('id', 'z', 'lm', 'lm_res', 'lsfr', 'mclump', 'fuclump', 'uclump', 'vclump',\
                                  'mfrac', 'fufrac', 'ufrac', 'vfrac', 'cc_sm', 'cc_sfr', 'cm_density', 'issfr', 'ossfr'),\
                                   meta={'name': 'cosmos clump id'})
    # ids_clumps_cat = Table([ids_clumps[:,i] for i in range(14)],\
    #                        names=('id', 'z', 'lm', 'lm_res', 'lsfr', 'mclump', 'fuclump', 'uclump', 'vclump', 'cc_sm', 'cc_sfr', 'cm_density', 'issfr', 'ossfr'),\
    #                                meta={'name': 'cosmos clump id'})
    ascii.write(ids_clumps_cat, '{}/clumps-catalog.dat'.format(dirname[:-5]), overwrite=True, format='commented_header')
