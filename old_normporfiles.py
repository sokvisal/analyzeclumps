from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, '../../scripts/')


import misc
import glob
import diagnostics


def caddnorm_plot(rgb_img, maps, normmaps, params, titleparams, res): # yi, xi, norm_r, norm_quan, uvmap
    sm, sfr, lu, lv, uvrest, binmap = maps
    massNorm = normmaps[0]
    Norm2800 = normmaps[1]
    uNorm = normmaps[2]
    vNorm = normmaps[3]
    xc, yc, ell_mh, ell_mf, ell_2800h, ell_2800f, ell_uh, ell_uf, ell_vh, ell_vf = params

    c = rgb_img.shape[0]/2
    citv = int(c*2/3)

    segrgb = rgb_img#[c-citv:c+citv,c-citv:c+citv]
    import matplotlib.pyplot as plt

    vmin=np.nanmin(uvrest)
    vmax=np.nanmax(uvrest)

    tmpx = np.arange(-0.5, 2.0, 0.1)
    tmpy = 0.06-1.6*tmpx-tmpx**2

    def map_clumps(normap, re, size):
        yi = normap[0]
        xi = normap[1]
        rnorm = normap[2]
        qnorm = normap[3]

        clump_map = np.zeros((size,size))

        inner_yx= []
        clump_yx= []
        outer_yx= []
        for i, (yval,xval) in enumerate(zip(qnorm, rnorm)):
            if xval < -0.5:#np.log10(0.3/(re*0.05)):#-0.5:
                inner_yx.append([yi[i], xi[i]])
            elif yval > 0.06-1.6*xval-xval**2:
                clump_yx.append([yi[i], xi[i]])
            elif yval < 0.06-1.6*xval-xval**2:
                outer_yx.append([yi[i], xi[i]])
        clump_yx = np.array(clump_yx)
        outer_yx = np.array(outer_yx)
        inner_yx = np.array(inner_yx)

        for i, data in enumerate([inner_yx, outer_yx, clump_yx]):
            if data.shape[0] > 0:
                clump_map[data[:,0], data[:,1]] = i+1
            # clump_map[outer_yx[:,0], outer_yx[:,1]] = 2
            # clump_map[inner_yx[:,0], inner_yx[:,1]] = 1

        return clump_map


    clumpmap_2800 = map_clumps(Norm2800, res[1], size=lu.shape[0])
    clumpmap_2800[clumpmap_2800==0] = np.nan
    clumpmap_u = map_clumps(uNorm, res[2], size=lu.shape[0])
    clumpmap_u[clumpmap_u==0] = np.nan
    clumpmap_v = map_clumps(vNorm, res[3], size=lu.shape[0])
    clumpmap_v[clumpmap_v==0] = np.nan
    clumpmap_m = map_clumps(massNorm, res[0], size=lu.shape[0])
    clumpmap_m[clumpmap_m==0] = np.nan

    clumpmap_2800_r = clumpmap_2800[~np.isnan(clumpmap_2800)].ravel()
    clumpmap_u_r = clumpmap_u[~np.isnan(clumpmap_u)].ravel()
    clumpmap_v_r = clumpmap_v[~np.isnan(clumpmap_v)].ravel()
    clumpmap_m_r = clumpmap_m[~np.isnan(clumpmap_m)].ravel()

    mravel = np.array(massNorm[5])
    ravel2800 = np.array(Norm2800[5])
    uravel = np.array(uNorm[5])#u[~np.isnan(u)].ravel()[uNorm[5]]
    vravel = np.array(vNorm[5])#v[~np.isnan(v)].ravel()[vNorm[5]]
    #
    # from scipy.stats import binned_statistic_2d
    # binx = np.arange(-3,1.1,.1)
    # tmpstat = binned_statistic_2d(runorm, unorm, uuv, 'median', bins=[binx,binx])
    #
    # plt.imshow(tmpstat.statistic.T, origin='lower', cmap='jet')
    # plt.show()


    # define extended clumps
    mclump = np.sum(mravel[clumpmap_m_r==3])/np.sum(mravel)
    clump2800 = np.sum(ravel2800[clumpmap_2800_r==3])/np.sum(ravel2800)
    uclump = np.sum(uravel[clumpmap_u_r==3])/np.sum(uravel)
    vclump = np.sum(vravel[clumpmap_v_r==3])/np.sum(vravel)

    clumpyid = [0,0,0,0]
    for i, fraction in enumerate([mclump, clump2800, uclump, vclump]):
        if 0.05<fraction<0.08:
            clumpyid[i] = 1
        elif fraction>=0.08:
            clumpyid[i] = 2

    cc_sm = clumps_contribution(uNorm[0], uNorm[1], sm, clumpmap_u_r)
    cc_sfr = clumps_contribution(uNorm[0], uNorm[1], sfr, clumpmap_u_r)
    # print 'Clump contribution in mass:, ', clumps_contribution(uNorm[0], uNorm[1], sm, clumpmap_u_r)
    # print 'Clump contribution in sfr:, ', cc_sfr
    clumpiness = np.sum(uravel[clumpmap_u_r==3])/np.sum(uravel)
    # print 'Clumps Fractional Contribution is L_v: ', np.sum(vravel[clumpmap_v_r==3])/np.sum(vravel)
    # print np.sum(uravel[clumpmap_u_r==3])/np.sum(uravel)
    # print np.sum(vravel[clumpmap_v_r==3])/np.sum(vravel)

    c = int(sm.shape[0]/2+2)
    citv = int(c*3./4)
    dc = c-citv

    segrgb = rgb_img[c-citv:c+citv,c-citv:c+citv]

    nrow = 4
    ncol = 4
    plt.rcParams.update({'font.size':6})
    fig = plt.figure(figsize=((ncol+1)*2.*3.4/4., (nrow+1)*2.))
    plt.show()
    #
    # from misc import rainbowb
    # from matplotlib import gridspec
    # from matplotlib.ticker import MaxNLocator
    # plt.rcParams.update({'font.size':11,\
    #                     'xtick.direction':'in', 'ytick.direction':'in', 'xtick.minor.visible':'True',\
    #                     'axes.linewidth':1.2,\
    #                     'xtick.major.width':1.2, 'xtick.minor.width':1.0, 'xtick.major.size':5., 'xtick.minor.size':3.0,\
    #                     'ytick.major.width':1.2, 'ytick.minor.width':1.0, 'ytick.major.size':5., 'ytick.minor.size':3.0})
    # gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0,\
    #                         top=1.-0.5/(nrow+1),bottom=0.5/(nrow+1),\
    #                         left=0.5/(ncol+1), right=1-0.5/(ncol+1), width_ratios=[1,1,0.4,1])
    #
    # rainbow = rainbowb()
    #
    # import matplotlib.colors
    # idl_rainbow = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkviolet", "blue", "cyan", "springgreen", "lime", "yellow", "orange", "red"])
    # cmap = matplotlib.colors.ListedColormap(['red', 'darkgrey', 'gold', 'blue'])
    # bounds=[0,1,2,3,4]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    #
    #
    # ################### RGB MAP ########################
    # ax = plt.subplot(gs[0,0], aspect='auto')
    # ax.imshow(segrgb, origin='lower')
    # ax.text(5,108,'(a)'.format(titleparams[0]), color='w', weight='bold')
    # ax.text(5,15,'z = {}'.format(titleparams[0]), color='w', weight='bold')
    # ax.text(5,5,'ID{}'.format(titleparams[1]), color='w', weight='bold')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.axis('off')
    #
    # # ################### (U-V)rest ########################
    # # ax = plt.subplot(gs[1], aspect='auto')
    # # ax.set_xticks([])
    # # ax.set_yticks([])
    # # ax.axis('off')
    #
    # ################### (U-V)rest ########################
    # ax = plt.subplot(gs[0,1], aspect='auto')
    # ax.imshow(uvrest[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # ax.text(5,108,'(e)', weight='bold')
    # ax.text(5,5,'Rest-frame (U-V) Color')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # ax.axis('off')
    #
    # ################### Stellar Mass ########################
    # ax = plt.subplot(gs[1,0], aspect='auto')
    # ax.imshow(sm[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='viridis')#, vmin=5.5)
    # ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="r")
    # ax.text(5,108,'(b)', weight='bold')
    # ax.text(5,5,'Stellar Mass')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.plot(xc-dc+ell_mh[0,:], yc-dc+ell_mh[1,:], linestyle='--',  color="r", linewidth=2, label='R$_\mathrm{e}$')
    # ax.plot(xc-dc+ell_mf[0,:], yc-dc+ell_mf[1,:], color="r", linewidth=2, label='2R$_\mathrm{e}$')
    # ax.legend(frameon=False, loc=1)
    #
    # ################### Mass Clump map ########################
    # ax = plt.subplot(gs[1,3], aspect='auto')
    # ax.imshow(clumpmap_m[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
    # # ax.text(5,5,'Mass Clump Map (%i)' %(int(extend_clumpyid[0])))
    # ax.text(5,108,'(i)', weight='bold')
    # ax.text(5,5,'Mass Clump Map')
    # if mclump>0.08: ax.text(5,15,'Mass Clumpy')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.plot(xc-dc+ell_mh[0,:], yc-dc+ell_mh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
    # ax.plot(xc-dc+ell_mf[0,:], yc-dc+ell_mf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')
    #
    # # ax.axis('off')
    #
    # ################### Stellar Mass Clumps ########################
    # ax = plt.subplot(gs[1,1], aspect='auto')
    # ax.scatter(massNorm[2], massNorm[3], c=massNorm[4], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # # ax.scatter(massNorm[2], massNorm[3], c=clumpmap_m_r, s=10, cmap=cmap, vmax=4)
    # ax.text(-1.5,-0.6,'Inner')
    # ax.text(-0.3,-1.6, 'Outer')
    # ax.text(-0.2,0.7, 'Clump')
    # ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
    # # ax.axvline(x=0, color='k', linestyle='-.', linewidth=1)
    # # ax.axvline(x=np.log10(2), color='k', linestyle='-.', linewidth=1)
    # ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
    # # ax.plot(tmpx, tmpy2, color='r', linestyle='-.', linewidth=1)
    #
    # ax.text(-2.3,0.875,'(f)', weight='bold')
    # ax.text(-2.3,-2., 'Stellar Mass')
    # ax.set_ylim([-2.2,1.2])
    # ax.set_xlim([-2.5,1.])
    # ax.set_aspect(3.5/3.4)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
    # ax.set_xlabel('log(R/R$_e$)')
    #
    # # labels = ax.get_yticklabels()
    # # # remove the first and the last labels
    # # labels[0] = labels[-1] = ""
    # # ax.set_yticklabels(labels)
    #
    # # ax.set_yticks([])
    # # ax.yaxis.set_major_locator(plt.NullLocator())
    # # ax.yaxis.set_major_locator(MaxNLocator(3))
    # # ax.xaxis.set_major_locator(MaxNLocator(5))
    #
    # ################### Urest Luminosity ########################
    # ax = plt.subplot(gs[2,0], aspect='auto')
    # ax.imshow(lu[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='inferno')
    # ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="tab:green")
    # ax.text(5,108,'(c)', weight='bold')
    # ax.text(5,5,'U$_\mathrm{rest}$')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.plot(xc-dc+ell_uh[0,:], yc-dc+ell_uh[1,:], linestyle='--',  color="tab:green", linewidth=2, label='R$_\mathrm{e}$')
    # ax.plot(xc-dc+ell_uf[0,:], yc-dc+ell_uf[1,:], color="tab:green", linewidth=2, label='2R$_\mathrm{e}$')
    # ax.legend(frameon=False, loc=1)
    # # ax.axis('off')
    #
    # ################### Clump map ########################
    # ax = plt.subplot(gs[2,3], aspect='auto')
    # ax.imshow(clumpmap_u[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
    # # ax.text(5,5,'U$_\mathrm{rest}$ Clump Map (%i)' %(int(extend_clumpyid[2])))
    # ax.text(5,108,'(j)', weight='bold')
    # ax.text(5,5,'U$_\mathrm{rest}$ Clump Map')
    # if uclump>0.08: ax.text(5,15,'U$_\mathrm{rest}$ Clumpy')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.plot(xc-dc+ell_uh[0,:], yc-dc+ell_uh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
    # ax.plot(xc-dc+ell_uf[0,:], yc-dc+ell_uf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')
    #
    #
    # ################### Urest Clump ########################
    # ax = plt.subplot(gs[2,1], aspect='auto') # aspect='equal'
    # ax.scatter(uNorm[2], uNorm[3], c=uNorm[4], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # # ax.scatter(uNorm[2], uNorm[3], c=clumpmap_u_r, s=10, cmap=cmap, vmax=4)
    # ax.text(-1.5,-0.6,'Inner')
    # ax.text(-0.3,-1.6, 'Outer')
    # ax.text(-0.2,0.7, 'Clump')
    # ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
    # # ax.axvline(x=0, color='k', linestyle='-.', linewidth=1)
    # # ax.axvline(x=np.log10(2), color='k', linestyle='-.', linewidth=1)
    # # ax.axhline(outerflux[2], color='k', linestyle='-.', linewidth=1)
    # # ax.axvline(x=np.log10(0.3/(res[1]*0.05)), color='k', linestyle='-', linewidth=1)
    # ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
    # # ax.plot(xcoord, ycoord, color='k', linestyle='--', linewidth=1)
    #
    # ax.text(-2.3,0.875,'(g)', weight='bold')
    # ax.text(-2.3,-2., 'U$_\mathrm{rest}$')
    # ax.set_ylim([-2.2,1.2])
    # ax.set_xlim([-2.5,1.])
    # ax.set_aspect(3.5/3.4)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
    # ax.set_xlabel('log(R/R$_e$)')
    # # ax.axis('off')
    #
    # # ax.set_yticks([])
    # # ax.yaxis.set_major_locator(plt.NullLocator())
    # # ax.yaxis.set_major_locator(MaxNLocator(5))
    # # ax.xaxis.set_major_locator(MaxNLocator(5))
    #
    # ################### Vrest Luminosity ########################
    # ax = plt.subplot(gs[3,0], aspect='auto')
    # ax.imshow(lv[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='inferno')
    # ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="tab:green")
    # ax.text(5,108,'(d)', weight='bold')
    # ax.text(5,5,'V$_\mathrm{rest}$')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.plot(xc-dc+ell_vh[0,:], yc-dc+ell_vh[1,:], linestyle='--',  color="tab:green", linewidth=2, label='R$_\mathrm{e}$')
    # ax.plot(xc-dc+ell_vf[0,:], yc-dc+ell_vf[1,:], color="tab:green", linewidth=2, label='2R$_\mathrm{e}$')
    # ax.legend(frameon=False, loc=1)
    # # ax.axis('off')
    #
    # ################### Clump map ########################
    # ax = plt.subplot(gs[3,3], aspect='auto')
    # ax.imshow(clumpmap_v[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
    # # ax.text(5,5,'V$_\mathrm{rest}$ Clump Map (%i)' %(int(extend_clumpyid[3])))
    # ax.text(5,108,'(k)', weight='bold')
    # ax.text(5,5,'V$_\mathrm{rest}$ Clump Map')
    # if vclump>0.08: ax.text(5,15,'V$_\mathrm{rest}$ Clumpy')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.plot(xc-dc+ell_vh[0,:], yc-dc+ell_vh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
    # ax.plot(xc-dc+ell_vf[0,:], yc-dc+ell_vf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')
    #
    # # ax.axis('off')
    #
    # ################### Urest Clump ########################
    # ax = plt.subplot(gs[3,1], aspect='auto') # aspect='equal'
    # # ax.scatter(vNorm[2], vNorm[3], c=vNorm[4], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
    # ax.scatter(vNorm[2], vNorm[3], c=clumpmap_v_r, s=10, cmap=cmap, vmax=4)
    # ax.text(-1.5,-0.6,'Inner')
    # ax.text(-0.3,-1.6, 'Outer')
    # ax.text(-0.2,0.7, 'Clump')
    #
    # ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
    # ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
    # ax.text(-2.3,0.875,'(h)', weight='bold')
    # ax.text(-2.3,-2., 'V$_\mathrm{rest}$')
    # ax.set_ylim([-2.2,1.2])
    # ax.set_xlim([-2.5,1.])
    # ax.set_aspect(3.5/3.4)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    # ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
    # ax.set_xlabel('log(R/R$_e$)')
    # plt.show()

    # if save is None: plt.show()
    # else:
    #     save.savefig(fig, dpi=300, bbox_inches = 'tight')
    #     plt.close()

    return fig, clumpyid, [clumpiness], [cc_sm, cc_sfr]

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

def coadd_profile(prop, phot_vars, zphot):
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
        import matplotlib.pyplot as plt
        from scipy.stats import mode

        r = np.sqrt((yi-yc)**2 + (xi-xc)**2)
        theta = np.rad2deg(np.arctan2((yi-yc), (xi-xc)))
        theta[theta<0] += 180.

        tbinsize = 5
        bintheta = np.arange(0,180+tbinsize,tbinsize)
        rbinsize = 2
        binr = np.arange(0,50+rbinsize,rbinsize)

        htheta = np.histogram(theta, bins = bintheta, density=True)
        hr = np.histogram(r, bins = binr, density=True)

        from scipy import interpolate
        from sklearn import mixture
        rs = np.arange(rbinsize/2., 50+rbinsize/2., rbinsize)
        ts = np.arange(tbinsize/2., 180+tbinsize/2., tbinsize)

        samples = np.dstack((rs,hr[0]))[0]

        def getbimodal(data, xbins, angle=False):
            from scipy import interpolate
            from sklearn import mixture

            def gauss_function(x, amp, x0, sigma):
                return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

            gmix = mixture.GaussianMixture(n_components = 2)
            fitted = gmix.fit(data) # data is shaped as (len, 1)

             # Construct function manually as sum of gaussians
            gmm_sum = np.full_like(xbins, fill_value=0, dtype=np.float32)
            for m, c, w in zip(fitted.means_.ravel(), fitted.covariances_.ravel(), fitted.weights_.ravel()):
                gauss = gauss_function(x=xbins, amp=1, x0=m, sigma=np.sqrt(c))
                gmm_sum += gauss / np.trapz(gauss, xbins) * w

                mindis = np.min([fitted.means_[0], fitted.means_[1]])
                maxdis = np.max([fitted.means_[0], fitted.means_[1]])
                if angle:
                    minidx = np.argmin(fitted.weights_)
                    mindis = fitted.means_[minidx][0]
                    maxidx = np.argmax(fitted.weights_)
                    maxdis = fitted.means_[maxidx][0]

            return mindis, maxdis, gmm_sum

        b, a, gmm_r_sum = getbimodal(r, rs)
        tmin, tmax, gmm_t_sum = getbimodal(theta, ts, angle=True)

#         from astropy.modeling import models, fitting
#         maxbid = np.where(htheta[0]==htheta[0].max())[0][0]
#         amplitude = htheta[0][maxbid]
#         maxmean = htheta[1][maxbid]

#         binfit = np.arange(tbinsize/2., 180+tbinsize/2., tbinsize)
#         g = models.Gaussian1D(amplitude, maxmean, 5.)
#         fitter = fitting.SLSQPLSQFitter()
#         g_fit = fitter(g, binfit, htheta[0])

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
        # return np.deg2rad(mode(theta.ravel().astype(int))[0][0]), a, b
        # return np.deg2rad(tmax), a, b
        return np.deg2rad(mode(theta.ravel().astype(int))[0][0]), max(r.ravel()), mode(r.ravel().astype(int))[0][0]

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

    import copy

    sm_unmask = copy.deepcopy(stellar_mass)

    #h = stellar_mass.shape[0]
    #w = stellar_mass.shape[0]

    to, a, b = galparams(stellar_mass)
    e = b/a
    # e = np.sqrt(1-b**2/a**2)
    # e = mwb/mwa
#     print 'gal params: ', to, mwb, mwa

    def halflightR(data, mass=False):
        '''
        Return normalized array wrt half-light/half-mass radius
        Set mass to True if working with half-mass radius
        '''
        summ = []
        npix = []

        if mass:
            maxr = a*0.4
        else:
            maxr = a*0.4 #int(a*2)
        # print data.shape, len(xi)
        for i in np.arange(1,maxr):
            # b = i*np.sqrt(1-e**2)
            # ell = ((xi-xc)*np.cos(to)+(yi-yc)*np.sin(to))**2./i**2 + ((xi-xc)*np.sin(to)-(yi-yc)*np.cos(to))**2./(b)**2.
            ell = ((xi-xc)*np.cos(to)+(yi-yc)*np.sin(to))**2./i**2 + ((xi-xc)*np.sin(to)-(yi-yc)*np.cos(to))**2./(i*e)**2.
            tmpidx = np.where(ell<1)[0]

            summ.append(sum(data[tmpidx]))
            npix.append(len(tmpidx))

        maxidx = np.argmax(summ)
        # print xc, yc, 116692, 117896, 118009
#         print abs(summ[:maxidx]-summ[maxidx]/2.)
        hidx = np.argmin(abs(summ[:maxidx]-summ[maxidx]/2.))
        qre = np.arange(1,maxr)[hidx]
        qnorm = summ[hidx]/npix[hidx]
        return qnorm, qre

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
        tmpidx = _re_cutoff(smask, 3*re)
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
        re2idx = _re_cutoff(dnorm, 3*re)

        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-m[~np.isnan(m)].ravel())
        ell_h, ell_f  = ellipses(re, e*re, to)
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
        re2idx = _re_cutoff(dnorm,  3*re)

        if not i:
            angdist = angular_distance(zphot)
            rkpc = (1./angdist)/0.05
            ssfr_i, ssfr_o = ssfr_ratio(m, lsfr, rkpc)
            mcd_re = cd(m, rkpc)

        sfrnorm = norm_sfrd(lsfr, re)
        ssfr = 10**(lsfr[~np.isnan(lsfr)].ravel()-m[~np.isnan(m)].ravel())
        ell_h, ell_f  = ellipses(re, e*re, to)
        tmp = mapNorm([yi, xi, rnorm, dnorm, uvrest[~np.isnan(m)].ravel(), photvar[~np.isnan(photvar)].ravel(), sfrnorm, agew[~np.isnan(agew)].ravel()], re2idx)

        gal_par.insert(len(gal_par), ell_h)
        gal_par.insert(len(gal_par), ell_f)
        norm_par.append(tmp)

    return masstot, gal_par, norm_par, [mcd_re, ssfr_i, ssfr_o], res
