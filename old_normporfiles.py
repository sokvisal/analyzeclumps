from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, '../../scripts/')


import misc
import glob
import diagnostics
from photutils import EllipticalAperture, EllipticalAnnulus
from photutils import aperture_photometry
from astropy.table import hstack

from misc import rainbowb
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.colors
import matplotlib.pyplot as plt

plt.rc('font', family='sans-serif')
plt.rcParams.update({'font.size':15,\
                    'xtick.direction':'in', 'ytick.direction':'in', 'xtick.minor.visible':'True',\
                    'axes.linewidth':1.2,\
                    'xtick.major.width':1.2, 'xtick.minor.width':1.0, 'xtick.major.size':5., 'xtick.minor.size':3.0,\
                    'ytick.major.width':1.2, 'ytick.minor.width':1.0, 'ytick.major.size':5., 'ytick.minor.size':3.0})

class Clump:
    def __init__(self, normap, shape):
        self.shape = shape

        self.yi = normap['yi']
        self.xi = normap['xi']
        self.rnorm = normap['rnorm']
        self.qnorm = normap['qnorm']
        self.umv = normap['umv']
        self.photvar = np.array( normap['quantity'] )

        self.clump_map, self.raveled_clump_map = self._create_map()
        self._create_map(y_int=-0.04)
        self._create_map(y_int=0.16)
        self._cal_clumpyFrac()

    def _create_map(self, y_int = 0.06):
        clump_map = np.zeros(self.shape)

        inner_yx= []
        clump_yx= []
        outer_yx= []
        for i, (yval,xval) in enumerate(zip(self.qnorm, self.rnorm)):
            if xval < -0.5:#np.log10(0.3/(re*0.05)):#-0.5:
                inner_yx.append([self.yi[i], self.xi[i]])
            elif yval > y_int - 1.6*xval-xval**2:
                clump_yx.append([self.yi[i], self.xi[i]])
            elif yval < y_int - 1.6*xval-xval**2:
                outer_yx.append([self.yi[i], self.xi[i]])
        clump_yx = np.array(clump_yx)
        outer_yx = np.array(outer_yx)
        inner_yx = np.array(inner_yx)

        for i, data in enumerate([inner_yx, outer_yx, clump_yx]):
            if data.shape[0] > 0:
                clump_map[data[:,0], data[:,1]] = i+1
            # clump_map[outer_yx[:,0], outer_yx[:,1]] = 2
            # clump_map[inner_yx[:,0], inner_yx[:,1]] = 1
        clump_map[clump_map==0] = np.nan

        if y_int == 0.06:
            return clump_map, clump_map[~np.isnan(clump_map)].ravel()
        elif y_int < 0.06:
            self.upplim_raveled_map = clump_map[~np.isnan(clump_map)].ravel()
        else:
            self.lowlim_raveled_map = clump_map[~np.isnan(clump_map)].ravel()

    def _cal_clumpyFrac(self):
        self.clumpFrac = np.sum(self.photvar[self.raveled_clump_map==3]) / np.sum(self.photvar)

    def get_map(self):
        return self.clump_map

    def get_ravel_map(self):
        return self.raveled_clump_map

    def get_rnorm(self):
        return self.rnorm

    def get_qnorm(self):
        return self.qnorm

    def get_uvm(self):
        return self.umv

    def get_clumpFrac(self):
        return self.clumpFrac

    def get_measured_photvar(self):
        sumClump = np.sum(self.photvar[self.raveled_clump_map==3])
        sum_lowlim = np.sum(self.photvar[self.lowlim_raveled_map==3])
        sum_upplim = np.sum(self.photvar[self.upplim_raveled_map==3])
        sumTot = np.sum(self.photvar)
        return sumClump, sum_lowlim, sum_upplim, sumTot


def caddnorm_plot(rgb_img, maps, normmaps, params, titleparams, plot=False, show=False, save=None): # yi, xi, norm_r, norm_quan, uvmap
    sm, sfr, lu, lv, uvrest, binmaps = maps
    # binmap = binmaps[0]
    # weighted_map = binmaps[1]

    massNorm = normmaps[0]
    Norm2800 = normmaps[1]
    uNorm = normmaps[2]
    vNorm = normmaps[3]

    cFUV = Clump(Norm2800, lu.shape)
    cUV = Clump(uNorm, lu.shape)
    cV = Clump(vNorm, lu.shape)
    cMass = Clump(massNorm, lu.shape)

    clumpiness = cUV.get_clumpFrac()
    uv_clump, uv_clump_low, uv_clump_upp, uv_gal = cUV.get_measured_photvar()

    cc_sm = clumps_contribution(uNorm['yi'], uNorm['xi'], sm, cUV.get_ravel_map() )
    cc_sfr = clumps_contribution(uNorm['yi'], uNorm['xi'], sfr, cUV.get_ravel_map() )

    clumpyid = [0,0,0,0]
    for i, fraction in enumerate([cMass.get_clumpFrac(), cFUV.get_clumpFrac(), cUV.get_clumpFrac(), cV.get_clumpFrac()]):
        if 0.05<fraction<0.08:
            clumpyid[i] = 1
        elif fraction>=0.08:
            clumpyid[i] = 2

    if plot:

        c = int(sm.shape[0]/2+2)
        citv = int(c*3./4)
        dc = c-citv

        segrgb = rgb_img[c-citv:c+citv,c-citv:c+citv]

        vmin=np.nanmin(uvrest)
        vmax=np.nanmax(uvrest)

        tmpx = np.arange(-0.5, 2.0, 0.1)
        tmpy = 0.06-1.6*tmpx-tmpx**2

        clumpmap_fuv = cFUV.get_map()
        clumpmap_uv = cUV.get_map()
        clumpmap_v = cV.get_map()
        clumpmap_m = cMass.get_map()
        xc, yc, to, a, b, ell_mh, ell_mf, ell_2800h, ell_2800f, ell_uh, ell_uf, ell_vh, ell_vf = params

        nrow = 4
        ncol = 4
        fig = plt.figure(figsize=((ncol+1)*2.*3.4/4., (nrow+1)*2.))

        gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0,\
                                top=1.-0.5/(nrow+1),bottom=0.5/(nrow+1),\
                                left=0.5/(ncol+1), right=1-0.5/(ncol+1), width_ratios=[1,1,0.4,1])

        rainbow = rainbowb()
        idl_rainbow = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkviolet", "blue", "cyan", "springgreen", "lime", "yellow", "orange", "red"])
        cmap = matplotlib.colors.ListedColormap(['red', 'darkgrey', 'gold', 'blue'])
        bounds=[0,1,2,3,4]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        ################### RGB MAP ########################
        ax = plt.subplot(gs[0,0], aspect='auto')
        ax.imshow(segrgb, origin='lower')
        ax.text(5,108,'(i)'.format(titleparams[0]), color='w', weight='bold')
        # ax.text(5,15,'z = {}'.format(titleparams[0]), color='w', weight='bold')
        ax.text(5,5,'ID{}'.format(titleparams[1]), color='w', weight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')

        # ################### (U-V)rest ########################
        # ax = plt.subplot(gs[1], aspect='auto')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.axis('off')

        ################### (U-V)rest ########################
        ax = plt.subplot(gs[0,1], aspect='auto')
        ax.imshow(uvrest[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='Spectral_r', vmin=vmin, vmax=vmax)
        ax.text(5,108,'(v)', weight='bold')
        ax.text(5,5,'Rest-frame (U-V)')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')

        ################## Stellar Mass ########################
        ax = plt.subplot(gs[1,0], aspect='auto')
        ax.imshow(sm[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='viridis')#, vmin=5.5)
        ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="r")
        ax.text(5,108,'(ii)', weight='bold')
        ax.text(5,5,'Mass')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_mh[0,:], yc-dc+ell_mh[1,:], linestyle='--',  color="r", linewidth=2, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_mf[0,:], yc-dc+ell_mf[1,:], color="r", linewidth=2, label='2R$_\mathrm{e}$')
        ax.legend(frameon=False, loc=1)

        ################## Stellar Mass Clumps ########################
        ax = plt.subplot(gs[1,1], aspect='auto')
        order = np.argsort(cMass.get_uvm())[::-1]
        ax.scatter(np.array(cMass.get_rnorm())[order], np.array(cMass.get_qnorm())[order], c=np.array(cMass.get_uvm())[order], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
        # ax.scatter(massNorm[2], massNorm[3], c=clumpmap_m_r, s=10, cmap=cmap, vmax=4)
        ax.text(-1.5,-0.6,'Inner')
        ax.text(-0.4,-1.8, 'Outer')
        ax.text(-0.4,0.7, 'Clump')
        ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
        # ax.axvline(x=0, color='k', linestyle='-.', linewidth=1)
        # ax.axvline(x=np.log10(2), color='k', linestyle='-.', linewidth=1)
        ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
        # ax.plot(tmpx, tmpy2, color='r', linestyle='-.', linewidth=1)

        ax.text(-2.3,0.875,'(vi)', weight='bold')
        ax.text(-2.3,-2., 'Mass')
        ax.set_ylim([-2.2,1.2])
        ax.set_xlim([-2.5,1.])
        ax.set_aspect(3.5/3.4)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
        ax.set_xlabel('log(R/R$_e$)')

        ################### Mass Clump map ########################
        ax = plt.subplot(gs[1,3], aspect='auto')
        ax.imshow(clumpmap_m[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
        # ax.text(5,5,'Mass Clump Map (%i)' %(int(extend_clumpyid[0])))
        ax.text(5,108,'(ix)', weight='bold')
        ax.text(5,5,'Mass Clump Map')
        if cMass.get_clumpFrac()>0.08: ax.text(5,15,'Mass Clumpy')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_mh[0,:], yc-dc+ell_mh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_mf[0,:], yc-dc+ell_mf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')


        ################### Urest Luminosity ########################
        ax = plt.subplot(gs[2,0], aspect='auto')
        ax.imshow(lu[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='inferno')
        ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="tab:green")
        ax.text(5,108,'(iii)', weight='bold')
        ax.text(5,5,'$U_\mathrm{rest}$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_uh[0,:], yc-dc+ell_uh[1,:], linestyle='--',  color="tab:green", linewidth=1, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_uf[0,:], yc-dc+ell_uf[1,:], color="tab:green", linewidth=1, label='2R$_\mathrm{e}$')
        ax.legend(frameon=False, loc=4)
        # ax.axis('off')

        ################### Urest Clump ########################
        ax = plt.subplot(gs[2,1], aspect='auto') # aspect='equal'
        order = np.argsort(cUV.get_uvm())#[::-1]
        ax.scatter(np.array(cUV.get_rnorm())[order], np.array(cUV.get_qnorm())[order], c=np.array(cUV.get_uvm())[order], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
        # ax.scatter(uNorm[2], uNorm[3], c=cUV.get_ravel_map(), s=10, cmap=cmap, vmax=4)
        ax.text(-1.5,-0.6,'Inner')
        ax.text(-0.4,-1.8, 'Outer')
        ax.text(-0.4,0.7, 'Clump')
        ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
        # ax.axvline(x=0, color='k', linestyle='-.', linewidth=1)
        # ax.axvline(x=np.log10(2), color='k', linestyle='-.', linewidth=1)
        # ax.axhline(outerflux[2], color='k', linestyle='-.', linewidth=1)
        # ax.axvline(x=np.log10(0.3/(res[1]*0.05)), color='k', linestyle='-', linewidth=1)
        ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
        # ax.plot(xcoord, ycoord, color='k', linestyle='--', linewidth=1)

        ax.text(-2.3,0.875,'(vii)', weight='bold')
        ax.text(-2.3,-2., '$U_\mathrm{rest}$')
        ax.set_ylim([-2.2,1.2])
        ax.set_xlim([-2.5,1.])
        ax.set_aspect(3.5/3.4)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
        ax.set_xlabel('log(R/R$_e$)')
        # ax.axis('off')

        # ax.set_yticks([])
        # ax.yaxis.set_major_locator(plt.NullLocator())
        # ax.yaxis.set_major_locator(MaxNLocator(5))
        # ax.xaxis.set_major_locator(MaxNLocator(5))

        ################### Clump map ########################
        ax = plt.subplot(gs[2,3], aspect='auto')
        ax.imshow(clumpmap_uv[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
        # ax.text(5,5,'U$_\mathrm{rest}$ Clump Map (%i)' %(int(extend_clumpyid[2])))
        ax.text(5,108,'(x)', weight='bold')
        ax.text(5,5,'$U_\mathrm{rest}$ Clump Map')
        if cUV.get_clumpFrac()>0.08: ax.text(5,20,'Clumpy in $U_\mathrm{rest}$')
        # if uclump>0.08: ax.text(5,15,'U$_\mathrm{rest}$ Clumpy')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_uh[0,:], yc-dc+ell_uh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_uf[0,:], yc-dc+ell_uf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')


        ################### Vrest Luminosity ########################
        ax = plt.subplot(gs[3,0], aspect='auto')
        ax.imshow(lv[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='inferno')
        ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="tab:green")
        ax.text(5,108,'(iv)', weight='bold')
        ax.text(5,5,'$V_\mathrm{rest}$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_vh[0,:], yc-dc+ell_vh[1,:], linestyle='--',  color="tab:green", linewidth=2, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_vf[0,:], yc-dc+ell_vf[1,:], color="tab:green", linewidth=2, label='2R$_\mathrm{e}$')
        ax.legend(frameon=False, loc=1)
        # ax.axis('off')

        ################### Urest Clump ########################
        ax = plt.subplot(gs[3,1], aspect='auto') # aspect='equal'
        ax.scatter(np.array(cV.get_rnorm()), np.array(cV.get_qnorm()), c=np.array(cV.get_uvm()), s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
        # ax.scatter(vNorm[2], vNorm[3], c=cV.get_ravel_map(), s=10, cmap=cmap, vmax=4)
        ax.text(-1.5,-0.6,'Inner')
        ax.text(-0.4,-1.8, 'Outer')
        ax.text(-0.4,0.7, 'Clump')

        ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
        ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
        ax.text(-2.3,0.875,'(viii)', weight='bold')
        ax.text(-2.3,-2., '$V_\mathrm{rest}$')
        ax.set_ylim([-2.2,1.2])
        ax.set_xlim([-2.5,1.])
        ax.set_aspect(3.5/3.4)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
        ax.set_xlabel('log(R/R$_e$)')

        ################### Clump map ########################
        ax = plt.subplot(gs[3,3], aspect='auto')
        ax.imshow(clumpmap_v[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
        # ax.text(5,5,'V$_\mathrm{rest}$ Clump Map (%i)' %(int(extend_clumpyid[3])))
        ax.text(5,108,'(xi)', weight='bold')
        ax.text(5,5,'$V_\mathrm{rest}$ Clump Map')
        if cV.get_clumpFrac()>0.08: ax.text(5,20,'Clumpy in $V_\mathrm{rest}$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_vh[0,:], yc-dc+ell_vh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_vf[0,:], yc-dc+ell_vf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')

        # ax.axis('off')

        savedir = '{}/_id-{}.png'.format(save, titleparams[1]) if save is not None else ''
        if show and save is not None:
            plt.savefig(savedir, dpi=300, bbox_inches = 'tight')
            plt.show()
        elif not show and save is not None:
            plt.savefig(savedir, dpi=300, bbox_inches = 'tight')
            plt.close()
        elif show and save is None:
            plt.show()
        else:
            plt.close()

        # elif show:
        #     plt.show()
        # else:
        #     plt.close()

        # if save is None: plt.show()
        # else:
        #     save.savefig(fig, dpi=300, bbox_inches = 'tight')
        #     plt.close()

    return clumpyid, [uv_clump, uv_clump_low, uv_clump_upp, uv_gal], [cc_sm, cc_sfr]

def clumps_contribution(yarray, xarray, map, clumps_map):
    # coords = zip(yarray, xarray)
    # y, x = zip(*coords)
    total = np.sum(10**map[yarray,xarray])

    idx = np.where(clumps_map==3)
    if len(idx) == 0:
        clumps_sum = 0.
    elif len(idx) == 1:
        cy = np.array(yarray).ravel()[idx]
        cx = np.array(xarray).ravel()[idx]
        clumps_sum = np.sum(10**map[cy,cx])
    else:
        cy = np.array(yarray).ravel()[idx]
        cx = np.array(xarray).ravel()[idx]

        clumps_coords = zip(cy, cx)
        y, x = zip(*clumps_coords)
        clumps_sum = np.sum(10**map[y,x])

    return clumps_sum/total


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
'''
Miscellaneous Functions needed to run the co-added normalized analysis
Needed to delete the useless functions and added some comments to others
'''

def _fluxd_to_flux(filename, binnumber, efflamdba):
    # efflambda in anstrom

    with open(filename) as f:
        data = f.readlines()[binnumber].split()[1:-1]

#     efflambda = np.array([4478.34, 4808.39, 6314.68, 5492.89, 9054.35, 16490.2, 12549.7, 10223.6]) # in angstrom

    flux = np.array([float(i) for i in data])

    flux[::2] *= (2.99e10)/(efflamdba*efflamdba*1e-8)
    flux[1::2] *= (2.99e10)/(efflamdba*efflamdba*1e-8)

    flux *= 10**(-73.6/2.5) # rescale from z=25 to z=-48.6
    # flux *= 1e-32
    # flux[:-3] /= 10**(-6.4/2.5)
    # flux[-3:] /= 10**(-5./2.5)

    return flux

def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def createEllMask(h, w, center=None, radius=None):

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

def _return_cm(mass):
    ############## Central Mass ######################
    c = mass.shape[0]/2
    cint = int(c/4)

    y, x = np.indices(mass.shape)
    y = y[int(c-cint):int(c+cint), int(c-cint):int(c+cint)]
    x = x[int(c-cint):int(c+cint), int(c-cint):int(c+cint)]

    m = 10**mass[int(c-cint):int(c+cint), int(c-cint):int(c+cint)].ravel()
    mcentral = 10**mass.ravel()
    idx = np.argwhere(~np.isnan(m))

    yi = y.ravel()[idx]
    xi = x.ravel()[idx]
    msel = m[idx]

    yc = np.sum(yi*msel)/np.sum(msel)
    xc = np.sum(xi*msel)/np.sum(msel)
    ##################################################

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

def coadd_profile(prop, phot_vars, zphot, weighted_map=None):
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

    yi, xi, yc, xc, pa, mwa, mwb = _return_cm(m)

    massmask = misc.createCircularMask(len(phot_vars[0]), len(phot_vars[0]), [xc, yc], 21)
    masstot = np.log10(np.nansum(10**(stellar_mass*massmask)))

    # import matplotlib.pyplot as plt
    # plt.imshow(massmask*stellar_mass)
    # plt.show()

    def galparams(stellar_mass, diagnostic=False):

        def myround(array, base=10):
            return [base * round(x/base) for x in array]
        import matplotlib.pyplot as plt
        from scipy.stats import mode

        r = np.sqrt((yi-yc)**2 + (xi-xc)**2)
        percentile =  np.sqrt((mode(myround(r.ravel(), 2))[0][0]**2)/(1-0.825**2)) #np.percentile(r.ravel(), 100)

        theta = np.rad2deg(np.arctan2((yi-yc), (xi-xc)))
        theta[theta<0] += 180.
        theta = theta[r.ravel()<percentile]

        tbinsize = 10
        bintheta = np.arange(0,180+tbinsize,tbinsize)
        rbinsize = 2
        binr = np.arange(0,50+rbinsize,rbinsize)

        if diagnostic:
            plt.hist(theta, bins = bintheta, density=True)
            plt.show()

            plt.hist(r[r.ravel()<percentile], bins = binr, density=True)
            plt.axvline(x= mode(myround(r.ravel(), 2))[0][0], color='tab:green')
            # plt.axvline(x=np.sqrt((mode(myround(r.ravel(), 2))[0][0]**2)/(1-0.64)))
            plt.axvline(x=np.sqrt((mode(myround(r.ravel(), 2))[0][0]**2)/(1-0.64)))
            plt.show()

        return np.deg2rad(mode(myround(theta.ravel()))[0][0]), max(r.ravel()[r.ravel()<percentile]), mode(myround(r.ravel(), 2))[0][0]

    def ellipses(a, b, to):
        # to = np.deg2rad(to)

        ells = []
        for i in np.arange(1,3):
            t = np.linspace(0, 2*np.pi, 100)
            Ell = np.array([i*a*np.cos(t), i*b*np.sin(t)])
                 #u,v removed to keep the same center location
            R_rot = np.array([[np.cos(to), -np.sin(to)],\
                              [np.sin(to), np.cos(to)]])
                 #2-D rotation matrix

            Ell_rot = np.zeros((2,Ell.shape[1]))
            for i in range(Ell.shape[1]):
                Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            ells.append(Ell_rot)
        return ells

    import copy

    sm_unmask = copy.deepcopy(stellar_mass)

    #h = stellar_mass.shape[0]
    #w = stellar_mass.shape[0]

    to, a, b = galparams(stellar_mass)
    e = b/a # axial ratio as defined by Wuyts12
    # a, b, phi = segmap_params
    # print (a,b)
    # e = np.sqrt(1-b**2/a**2)
    # to = phi

    # print ('eccentricity', e, b/a)
    # e = mwb/mwa
#     print 'gal params: ', to, mwb, mwa
    import matplotlib.pyplot as plt
    def halflightR(data, img, mass=False, weighted_map=None):
        '''
        Return normalized array wrt half-light/half-mass radius
        Set mass to True if working with half-mass radius
        '''

        data = img.ravel()
        tmpy, tmpx = np.indices(img.shape)
        tmpy = np.ravel(tmpy)
        tmpx = np.ravel(tmpx)


        radii = np.arange(0.5, a, 0.5)
        # b = radii*np.sqrt(1-e**2)
        # ell = ((tmpx-xc)*np.cos(to)+(tmpy-yc)*np.sin(to))**2./a**2 + ((tmpx-xc)*np.sin(to)-(tmpy-yc)*np.cos(to))**2./(a*e)**2.
        # tmpidx = np.where(ell<1)[0]
        # maxflux = np.nansum(data[tmpidx])

        apertures = [ EllipticalAperture((xc,yc), r, r*e, theta=to) for r in radii ]
        phot_table = aperture_photometry(img, apertures, mask=np.isnan(img) )
        apertureNames = phot_table.colnames[3:]
        # print (np.lib.recfunctions.structured_to_unstructured( phot_table[apertureNames].as_array()[0] ))
        apertureFluxes  = [ phot_table[apertureName].data[0] for apertureName in apertureNames ]
        apertureAreas =  [ aperture.area for aperture in apertures ]

        if weighted_map is not None:
            weighted_phot_table = aperture_photometry(img/weighted_map, apertures, mask=np.isnan(img) )
            weightedFluxes  =  [ weighted_phot_table[apertureName].data[0] for apertureName in apertureNames ]

            # maxidx = np.argmax(weightedFluxes[:]) #tmpidx
            tmpidx = np.argmin(abs(weightedFluxes - weightedFluxes[ np.argmax(weightedFluxes[:]) ]/2.))
            weighted_qre = radii[tmpidx]
            weighted_qnorm = apertureFluxes[tmpidx]/apertureAreas[tmpidx]

        # radii = np.arange(0.5, a, 0.5)
        # phot_table = hstack([ aperture_photometry(img/weighted_map, EllipticalAnnulus((xc,yc), r, r+1, (r+1)*np.sqrt(1-e**2), theta=to), mask=np.isnan(img) ) for r in radii ])
        # annulusFluxes = np.array([ phot_table['aperture_sum_{}'.format(i+1)].data[0] for i in range(len(radii))])
        # annulusAreas =  np.array([ EllipticalAnnulus((xc,yc), r, r+1, (r+1)*e, theta=to).area for r in radii])

        # tmpidx = np.where(radii<a*0.5)[0][-1]
        maxidx = np.argmax(apertureFluxes[:]) #tmpidx
        hidx = np.argmin(abs(apertureFluxes[:maxidx]-apertureFluxes[maxidx]/2.))
        qre = radii[hidx]
        qnorm = apertureFluxes[hidx]/apertureAreas[hidx]

        # fig = plt.figure(figsize=(6,4))
        # ax = fig.add_subplot(1,1,1)
        # ax2 = ax.twinx()
        #
        # ax.scatter(radii, apertureFluxes, color='tab:blue')
        # ax.scatter(radii, weightedFluxes, color='tab:red')
        # # ax.axvline(x=maxidx)
        # ax.axvline(x=qre, color='grey')
        # ax.axvline(x=weighted_qre, linestyle='--', color='grey')
        # # ax.axhline(y=summ[maxidx]/2., linestyle=':', color='grey')
        #
        # # ax2.scatter(radii, annulusFluxes, color='tab:purple')
        # ax2.scatter(radii, annulusFluxes/annulusAreas, color='tab:purple')
        # ax2.set_ylim([-max(annulusFluxes/annulusAreas)*0.1, max(annulusFluxes/annulusAreas)*1.1])
        # plt.show()

        return weighted_qnorm, weighted_qre#qnorm, qre


    def _re_cutoff(array, qre):
        ell = ((xi-xc)*np.cos(to)+(yi-yc)*np.sin(to))**2./(qre)**2 \
                + ((xi-xc)*np.sin(to)-(yi-yc)*np.cos(to))**2./(qre*e)**2.
        tmpidx = np.where(ell<1)[0]
        return tmpidx

    # mravel = m.ravel()
    # mmask = 10**(m[~np.isnan(m)].ravel())
    # mnorm, mre = halflightR(mmask, mass=True)

    r = np.sqrt((yi-yc)**2+(xi-xc)**2).ravel()

    def mapNorm(ndarray, masked_coord):
        ''' ndarray: yi, xi, norm_r, norm_quan, uvmap '''
        keys = ['yi', 'xi', 'rnorm', 'qnorm', 'umv', 'quantity']

        assert len(ndarray) == len(keys), "length of input ndarray must be length of keys"
        if masked_coord is None:
            return {key: list(d) for (key, d) in zip(keys, ndarray)}
        else:
            return {key: list(d[masked_coord]) for (key, d) in zip(keys, ndarray)}

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
        tmpidx = _re_cutoff(smask, 3*re)
        newidx = np.array([x for x in tmpidx if not x in idx])
        # print idx, newidx

        ssfr_i = np.sum(smask[idx])/np.sum(mmask[idx])
        ssfr_o = np.sum(smask[newidx])/np.sum(mmask[newidx])

        return ssfr_i, ssfr_o

    gal_par = [xc, yc, to, a, b]
    norm_par = []
    for i, physvar in enumerate(prop[:1]):
        dmask = 10**(physvar[~np.isnan(physvar)].ravel())
        norm, re = halflightR(dmask, 10**physvar, mass=True, weighted_map=weighted_map)

        dnorm = np.log10(dmask/norm)
        rnorm = np.log10(r/re)
        re2idx = _re_cutoff(dnorm, 3*re)

        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-m[~np.isnan(m)].ravel())
        ell_h, ell_f  = ellipses(re, e*re, to)
        tmp = mapNorm([yi, xi, rnorm, dnorm, uvrest[~np.isnan(m)].ravel(), physvar[~np.isnan(physvar)].ravel()], re2idx ) #, sfrnorm, agew[~np.isnan(agew)].ravel()], re2idx)

        gal_par.insert(len(gal_par), ell_h)
        gal_par.insert(len(gal_par), ell_f)
        norm_par.append(tmp) #[list(dnorm), list(rnorm)])


    import matplotlib.pyplot as plt
    for i, photvar in enumerate(phot_vars):
        dmask = photvar[~np.isnan(photvar)].ravel()

        norm, re = halflightR(dmask, photvar, mass=False, weighted_map=weighted_map)

        dnorm = np.log10(dmask/norm)
        rnorm = np.log10(r/re)
        re2idx = _re_cutoff(dnorm,  3*re)

        if not i:
            angdist = angular_distance(zphot)
            rkpc = (1./angdist)/0.05
            ssfr_i, ssfr_o = ssfr_ratio(m, lsfr, rkpc)
            mcd_re = cd(m, rkpc)

        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-m[~np.isnan(m)].ravel())
        ell_h, ell_f  = ellipses(re, e*re, to)
        tmp = mapNorm([yi, xi, rnorm, dnorm, uvrest[~np.isnan(m)].ravel(), photvar[~np.isnan(photvar)].ravel()], re2idx ) #, sfrnorm, agew[~np.isnan(agew)].ravel()], re2idx)

        gal_par.insert(len(gal_par), ell_h)
        gal_par.insert(len(gal_par), ell_f)
        norm_par.append(tmp)

    return masstot, gal_par, norm_par, [mcd_re, ssfr_i, ssfr_o]
