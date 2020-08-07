
import numpy as np
import matplotlib.pyplot as plt
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy import interpolate
import os

import matplotlib
# matplotlib.use('Agg')


def make_profile(rgb_img, maps, normmaps, params, titleparams, tile, outerflux=[None, None, None, None], res=[None, None, None, None], savedir=False, showplot=True): # yi, xi, norm_r, norm_quan, uvmap
    sm, sfr, lu, lv, uvrest, binmap = maps
    massNorm = normmaps[0]
    Norm2800 = normmaps[1]
    uNorm = normmaps[2]
    vNorm = normmaps[3]
    xc, yc, ell_mh, ell_mf, ell_2800h, ell_2800f, ell_uh, ell_uf, ell_vh, ell_vf = params

    vmin=-0.25#np.nanmin(uvrest)
    vmax=2.5#np.nanmax(uvrest)

    tmpx = np.arange(-0.5, 2.0, 0.1)
    slope = 1.
    xshift = -0.8
    shift = 0.1
    yshift = 0.7
    tmpy = -(tmpx/slope-xshift)**2+yshift #0.06-1.6*tmpx-tmpx**2
    tmpy2 = -(tmpx/slope-xshift-shift)**2+yshift

    def _get_massprofile(norm_massmap):
        yi = np.array(norm_massmap[3])
        xi = np.array(norm_massmap[2])

        def func(x):
            return -(x/slope-xshift)**2+yshift

        dx = 0.05
        xrange = np.arange(-0.7,0.6,dx)

        coord = []
        for x in xrange[:-1]:
            tmpx = xi[(xi>x)&(xi<=x+dx)]
            tmpy = yi[(xi>x)&(xi<=x+dx)]

            if len(tmpx)==0:
                coord.append([func(x+dx/2.),x+dx/2.])
            else:
                idx = np.argmax(tmpy)
                if tmpy[idx] < func(tmpx[idx]):
                    coord.append([func(x+dx/2.),x+dx/2.])
                else:
                    coord.append([tmpy[idx],tmpx[idx]])
        coord = np.array(coord)

        y = coord[:,0]
        x = coord[:,1]
        func = interpolate.interp1d(x, y)
        xnew = np.arange(-0.5, 0.5, 0.01)
        ynew = func(xnew)
        return ynew, xnew

    def _get_massprofile_new(norm_massmap):
        yi = np.array(norm_massmap[3])
        xi = np.array(norm_massmap[2])
        print (len(yi), len(xi))
        x = xi[xi>-0.5]
        y = yi[xi>-0.5]

        shift = 0.
        while True:
            tmpy = -(x/slope-xshift-shift)**2+yshift+shift
            if np.min(tmpy-y) < 0:
                shift += 0.01
            else: break
        return shift

    ycoord, xcoord = _get_massprofile(massNorm)
    # shift = _get_massprofile_new(massNorm)
    # mprofile = -(tmpx/slope-xshift-shift)**2+yshift+shift

    def map_clumps(normap, norm_anuflux, re, size, mass=False):
        yi = normap[0]
        xi = normap[1]
        rnorm = normap[2]
        qnorm = normap[3]
        agew = normap[7]

        clump_map = np.zeros((size,size))

        inner_yx= []
        clump_yx= []
        compact_clump_yx=[]
        outer_yx= []

        clumpy_regions = []
        for i, (yval,xval) in enumerate(zip(qnorm, rnorm)):
            if xval < -0.5:#np.log10(0.3/(re*0.05)):#-0.5:
                inner_yx.append([yi[i], xi[i]])
            elif yval > -(xval/slope-xshift)**2+yshift: #0.06-1.6*xval-xval**2: #yval > 0.06-1.6*xval-xval**2 and yval > norm_anuflux:
                clump_yx.append([yi[i], xi[i]])
                clumpy_regions.append([xval-np.sqrt(0.7-yval)+0.8, xval]) #[yval+(xval/slope-xshift)**2-yshift,xval]
                # compact_clump_yx.append([np])
                # tmpidx = np.nanargmin(abs(xval-xcoord))
                # if yval>-(xval/slope-xshift-shift)**2+yshift:
                #     compact_clump_yx.append([yi[i], xi[i]])
            else:
                clumpy_regions.append([np.nan, np.nan])
                outer_yx.append([yi[i], xi[i]])
        clump_yx = np.array(clump_yx)
        outer_yx = np.array(outer_yx)
        inner_yx = np.array(inner_yx)

        clumpy_regions = np.array(clumpy_regions)

        # tmpidx = np.where(clumpy_regions[:,0]>np.median(clumpy_regions[:,0]))
        compact_clump_yx = np.array(compact_clump_yx)
        # print np.std(clumpy_regions[:,0])
        # plt.scatter(clumpy_regions[:,1], clumpy_regions[:,0])
        # plt.show()
        for i, data in enumerate([inner_yx, outer_yx, clump_yx, compact_clump_yx]):
            if data.shape[0] > 0:
                clump_map[data[:,0], data[:,1]] = i+1
        return clump_map, None # clump_agew_r_info

    clumpmap_2800, tmp  = map_clumps(Norm2800, outerflux[1], res[1], size=lu.shape[0])
    clumpmap_2800[clumpmap_2800==0] = np.nan
    clumpmap_u, tmp = map_clumps(uNorm, outerflux[2],  res[2], size=lu.shape[0])
    clumpmap_u[clumpmap_u==0] = np.nan
    clumpmap_v, clump_agew_r_info = map_clumps(vNorm, outerflux[3], res[3], size=lu.shape[0])
    clumpmap_v[clumpmap_v==0] = np.nan
    clumpmap_m, tmp = map_clumps(massNorm, outerflux[0], res[0], size=lu.shape[0])
    clumpmap_m[clumpmap_m==0] = np.nan

    clumpmap_2800_r = clumpmap_2800[~np.isnan(clumpmap_2800)].ravel()
    clumpmap_u_r = clumpmap_u[~np.isnan(clumpmap_u)].ravel()
    clumpmap_v_r = clumpmap_v[~np.isnan(clumpmap_v)].ravel()
    clumpmap_m_r = clumpmap_m[~np.isnan(clumpmap_m)].ravel()


    def _find_clumpregions(clumpmap, clumpid):
        tmpmap = clumpmap.copy()

        tmpmap[np.isnan(tmpmap)] = 0.
        tmpmap[tmpmap<clumpid] = 0.
        tmpmap[tmpmap==clumpid] = 1.

        distance = ndi.distance_transform_edt(tmpmap)
        local_maxi = peak_local_max(distance, min_distance=4, indices=False, labels=tmpmap)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=tmpmap)
        for i in range(1,np.max(labels)+1):
            if len(labels[labels==i].ravel()) < 12:
                clumpmap[labels==i] = clumpid-1

        # plt.imshow(-distance, origin='lower')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(clumpmap, cmap='Spectral_r', origin='lower', vmax=3)
        # plt.colorbar()
        # plt.show()
        return clumpmap

    clumpmaps = [clumpmap_m, clumpmap_2800, clumpmap_u, clumpmap_v]
    clumpmaps_raveled = [clumpmap_m_r, clumpmap_2800_r, clumpmap_u_r, clumpmap_v_r]
    for i, map in enumerate(clumpmaps):
        # clumpmaps[i] = _find_clumpregions(map, clumpid=4)
        # clumpmaps[i] = _find_clumpregions(clumpmaps[i], clumpid=3)
        clumpmaps_raveled[i] = _ravel_nonnan_map(clumpmaps[i])
    clumpmap_m, clumpmap_2800, clumpmap_u, clumpmap_v = clumpmaps
    clumpmap_m_r, clumpmap_2800_r, clumpmap_u_r, clumpmap_v_r = clumpmaps_raveled

    mravel = 10**np.array(massNorm[5])
    ravel2800 = np.array(Norm2800[5])
    uravel = np.array(uNorm[5])#u[~np.isnan(u)].ravel()[uNorm[5]]
    vravel = np.array(vNorm[5])#v[~np.isnan(v)].ravel()[vNorm[5]]

    # define compact clumps as compact
    mcompact = np.sum(mravel[clumpmap_m_r==4])/np.sum(mravel)
    fucompact = np.sum(ravel2800[clumpmap_2800_r==4])/np.sum(ravel2800)
    ucompact = np.sum(uravel[clumpmap_u_r==4])/np.sum(uravel)
    vcompact = np.sum(vravel[clumpmap_v_r==4])/np.sum(vravel)

    compact_clumpyid = [0,0,0,0]
    for i, fraction in enumerate([mcompact, fucompact, ucompact, vcompact]):
        if 0.05<fraction<0.08:
            compact_clumpyid[i] = 1
        elif fraction>=0.08:
            compact_clumpyid[i] = 2

    # define extended clumps
    mextend = np.sum(mravel[clumpmap_m_r==3])/np.sum(mravel)
    fuextend = np.sum(ravel2800[clumpmap_2800_r==3])/np.sum(ravel2800)
    uextend = np.sum(uravel[clumpmap_u_r==3])/np.sum(uravel)
    vextend = np.sum(vravel[clumpmap_v_r==3])/np.sum(vravel)

    extend_clumpyid = [0,0,0,0]
    for i, fraction in enumerate([mextend, fuextend, uextend, vextend]):
        if 0.05<fraction<0.08:
            extend_clumpyid[i] = 1
        elif fraction>=0.08:
            extend_clumpyid[i] = 2

    # print ('Exlended Clump ID: ', extend_clumpyid)
    # print ('Compact Clump ID: ', compact_clumpyid)

    if showplot:
        c = int(sm.shape[0]/2+2)
        citv = int(c*3./4)
        dc = c-citv

        segrgb = rgb_img[c-citv:c+citv,c-citv:c+citv]

        nrow = 4
        ncol = 4
        plt.rcParams.update({'font.size':6})
        fig = plt.figure(figsize=((ncol+1)*2.*3.4/4., (nrow+1)*2.))

        from misc import rainbowb
        from matplotlib import gridspec
        from matplotlib.ticker import MaxNLocator
        plt.rcParams.update({'font.size':11,\
                            'xtick.direction':'in', 'ytick.direction':'in', 'xtick.minor.visible':'True',\
                            'axes.linewidth':1.2,\
                            'xtick.major.width':1.2, 'xtick.minor.width':1.0, 'xtick.major.size':5., 'xtick.minor.size':3.0,\
                            'ytick.major.width':1.2, 'ytick.minor.width':1.0, 'ytick.major.size':5., 'ytick.minor.size':3.0})
        gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0,\
                                top=1.-0.5/(nrow+1),bottom=0.5/(nrow+1),\
                                left=0.5/(ncol+1), right=1-0.5/(ncol+1), width_ratios=[1,1,0.4,1])

        rainbow = rainbowb()

        import matplotlib.colors
        idl_rainbow = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "darkviolet", "blue", "cyan", "springgreen", "lime", "yellow", "orange", "red"])
        cmap = matplotlib.colors.ListedColormap(['red', 'darkgrey', 'gold', 'blue'])
        bounds=[0,1,2,3,4]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)


        ################### RGB MAP ########################
        ax = plt.subplot(gs[0,0], aspect='auto')
        ax.imshow(segrgb, origin='lower')
        ax.text(5,108,'(a)'.format(titleparams[0]), color='w', weight='bold')
        ax.text(5,15,'z = {}'.format(titleparams[0]), color='w', weight='bold')
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
        ax.text(5,108,'(e)', weight='bold')
        ax.text(5,5,'Rest-frame (U-V) Color')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')

        ################### Stellar Mass ########################
        ax = plt.subplot(gs[1,0], aspect='auto')
        ax.imshow(sm[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='viridis')#, vmin=5.5)
        ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="r")
        ax.text(5,108,'(b)', weight='bold')
        ax.text(5,5,'Stellar Mass')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_mh[0,:], yc-dc+ell_mh[1,:], linestyle='--',  color="r", linewidth=2, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_mf[0,:], yc-dc+ell_mf[1,:], color="r", linewidth=2, label='2R$_\mathrm{e}$')
        ax.legend(frameon=False, loc=1)

        ################### Mass Clump map ########################
        ax = plt.subplot(gs[1,3], aspect='auto')
        ax.imshow(clumpmap_m[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
        # ax.text(5,5,'Mass Clump Map (%i)' %(int(extend_clumpyid[0])))
        ax.text(5,108,'(i)', weight='bold')
        ax.text(5,5,'Mass Clump Map')
        if mextend>0.08: ax.text(5,15,'Mass Clumpy')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_mh[0,:], yc-dc+ell_mh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_mf[0,:], yc-dc+ell_mf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')

        # ax.axis('off')

        ################### Stellar Mass Clumps ########################
        ax = plt.subplot(gs[1,1], aspect='auto')
        ax.scatter(massNorm[2], massNorm[3], c=massNorm[4], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
        # ax.scatter(massNorm[2], massNorm[3], c=clumpmap_m_r, s=10, cmap=cmap, vmax=4)
        ax.text(-1.5,-0.6,'Inner')
        ax.text(-0.3,-1.6, 'Outer')
        ax.text(-0.2,0.7, 'Clump')
        ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
        # ax.axvline(x=0, color='k', linestyle='-.', linewidth=1)
        # ax.axvline(x=np.log10(2), color='k', linestyle='-.', linewidth=1)
        ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
        # ax.plot(tmpx, tmpy2, color='r', linestyle='-.', linewidth=1)

        ax.text(-2.3,0.875,'(f)', weight='bold')
        ax.text(-2.3,-2., 'Stellar Mass')
        ax.set_ylim([-2.2,1.2])
        ax.set_xlim([-2.5,1.])
        ax.set_aspect(3.5/3.4)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
        ax.set_xlabel('log(R/R$_e$)')

        # labels = ax.get_yticklabels()
        # # remove the first and the last labels
        # labels[0] = labels[-1] = ""
        # ax.set_yticklabels(labels)

        # ax.set_yticks([])
        # ax.yaxis.set_major_locator(plt.NullLocator())
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        # ax.xaxis.set_major_locator(MaxNLocator(5))

        ################### Urest Luminosity ########################
        ax = plt.subplot(gs[2,0], aspect='auto')
        ax.imshow(lu[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='inferno')
        ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="tab:green")
        ax.text(5,108,'(c)', weight='bold')
        ax.text(5,5,'U$_\mathrm{rest}$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_uh[0,:], yc-dc+ell_uh[1,:], linestyle='--',  color="tab:green", linewidth=2, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_uf[0,:], yc-dc+ell_uf[1,:], color="tab:green", linewidth=2, label='2R$_\mathrm{e}$')
        ax.legend(frameon=False, loc=1)
        # ax.axis('off')

        ################### Clump map ########################
        ax = plt.subplot(gs[2,3], aspect='auto')
        ax.imshow(clumpmap_u[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
        # ax.text(5,5,'U$_\mathrm{rest}$ Clump Map (%i)' %(int(extend_clumpyid[2])))
        ax.text(5,108,'(j)', weight='bold')
        ax.text(5,5,'U$_\mathrm{rest}$ Clump Map')
        if uextend>0.08: ax.text(5,15,'U$_\mathrm{rest}$ Clumpy')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_uh[0,:], yc-dc+ell_uh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_uf[0,:], yc-dc+ell_uf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')


        ################### Urest Clump ########################
        ax = plt.subplot(gs[2,1], aspect='auto') # aspect='equal'
        ax.scatter(uNorm[2], uNorm[3], c=uNorm[4], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
        # ax.scatter(uNorm[2], uNorm[3], c=clumpmap_u_r, s=10, cmap=cmap, vmax=4)
        ax.text(-1.5,-0.6,'Inner')
        ax.text(-0.3,-1.6, 'Outer')
        ax.text(-0.2,0.7, 'Clump')
        ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
        # ax.axvline(x=0, color='k', linestyle='-.', linewidth=1)
        # ax.axvline(x=np.log10(2), color='k', linestyle='-.', linewidth=1)
        # ax.axhline(outerflux[2], color='k', linestyle='-.', linewidth=1)
        # ax.axvline(x=np.log10(0.3/(res[1]*0.05)), color='k', linestyle='-', linewidth=1)
        ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
        # ax.plot(xcoord, ycoord, color='k', linestyle='--', linewidth=1)

        ax.text(-2.3,0.875,'(g)', weight='bold')
        ax.text(-2.3,-2., 'U$_\mathrm{rest}$')
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

        ################### Vrest Luminosity ########################
        ax = plt.subplot(gs[3,0], aspect='auto')
        ax.imshow(lv[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap='inferno')
        ax.plot([xc-dc], [yc-dc], marker='x', markersize=6, color="tab:green")
        ax.text(5,108,'(d)', weight='bold')
        ax.text(5,5,'V$_\mathrm{rest}$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_vh[0,:], yc-dc+ell_vh[1,:], linestyle='--',  color="tab:green", linewidth=2, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_vf[0,:], yc-dc+ell_vf[1,:], color="tab:green", linewidth=2, label='2R$_\mathrm{e}$')
        ax.legend(frameon=False, loc=1)
        # ax.axis('off')

        ################### Clump map ########################
        ax = plt.subplot(gs[3,3], aspect='auto')
        ax.imshow(clumpmap_v[c-citv:c+citv,c-citv:c+citv], origin='lower', cmap=cmap, vmax=4)
        # ax.text(5,5,'V$_\mathrm{rest}$ Clump Map (%i)' %(int(extend_clumpyid[3])))
        ax.text(5,108,'(k)', weight='bold')
        ax.text(5,5,'V$_\mathrm{rest}$ Clump Map')
        if vextend>0.08: ax.text(5,15,'V$_\mathrm{rest}$ Clumpy')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xc-dc+ell_vh[0,:], yc-dc+ell_vh[1,:], linestyle='--',  color="k", linewidth=1, label='R$_\mathrm{e}$')
        ax.plot(xc-dc+ell_vf[0,:], yc-dc+ell_vf[1,:], color="k", linewidth=1, label='2R$_\mathrm{e}$')

        # ax.axis('off')

        ################### Urest Clump ########################
        ax = plt.subplot(gs[3,1], aspect='auto') # aspect='equal'
        # ax.scatter(vNorm[2], vNorm[3], c=vNorm[4], s=10, cmap='Spectral_r', vmin=vmin, vmax=vmax)
        ax.scatter(vNorm[2], vNorm[3], c=clumpmap_v_r, s=10, cmap=cmap, vmax=4)
        ax.text(-1.5,-0.6,'Inner')
        ax.text(-0.3,-1.6, 'Outer')
        ax.text(-0.2,0.7, 'Clump')

        ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
        ax.plot(tmpx, tmpy, color='k', linestyle='--', linewidth=2)
        ax.text(-2.3,0.875,'(h)', weight='bold')
        ax.text(-2.3,-2., 'V$_\mathrm{rest}$')
        ax.set_ylim([-2.2,1.2])
        ax.set_xlim([-2.5,1.])
        ax.set_aspect(3.5/3.4)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
        ax.set_xlabel('log(R/R$_e$)')

        # ax = plt.subplot(gs[11], aspect='auto') # aspect='equal'
        # ax.scatter(uNorm[2], uNorm[3], c=clumpmap_u_r, s=10, cmap=cmap, vmax=4)
        # # plt.plot(xcoord, ycoord, color='k', linestyle='--', linewidth=2)
        # ax.axvline(x=-0.5, color='k', linestyle='--', linewidth=2)
        # # ax.axvline(x=0, color='k', linestyle='-.', linewidth=1)
        # # ax.axvline(x=np.log10(2), color='k', linestyle='-.', linewidth=1)
        # ax.plot(tmpx, tmpy, color='k', linestyle='-.', linewidth=1)
        # ax.text(-1.5,-1.5,'Inner')
        # ax.text(-0.3,-2., 'Outer')
        # ax.text(-0.2,0.5, 'Clump')
        # ax.set_ylim([-2.5,1.])
        # ax.set_xlim([-2.5,1.])
        # ax.set_aspect(3.5/3.5)
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        # ax.set_ylabel(r'log($\Sigma/\Sigma_{e}$)')
        # ax.set_xlabel('log(R/R$_e$)')
        # ax.axis('off')

        if not os.path.isdir('{}'.format(savedir)):
            os.makedirs('{}'.format(savedir))

        if savedir:
            plt.savefig('{}/_id-{}.jpg'.format(savedir, titleparams[1]), bbox_inches='tight',dpi=300)
            # print './tmpplots/_id-{}.jpg'.format(titleparams[1])
            plt.close()
        elif showplot and not savedir:
            plt.show()
            plt.close()
        else:
            plt.close()

    # nrow = 1
    # ncol = 4
    # plt.rcParams.update({'font.size':6})
    # fig = plt.figure(figsize=(ncol*3, nrow*3))
    # plt.rcParams.update({'font.size':12})
    # plt.rc('axes', labelsize=12)
    # gs = gridspec.GridSpec(nrow, ncol, wspace=0.05, hspace=0.05)
    #
    # for i, clumpmap in enumerate(clumpmaps):
    #     ax = plt.subplot(gs[i])
    #     clumpmap[clumpmap<=2] = np.nan
    #     ax.imshow(clumpmap, origin='lower', cmap='prism')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # plt.show()


    cc_sm = clumps_contribution(uNorm[0], uNorm[1], sm, clumpmap_u_r)
    cc_sfr = clumps_contribution(uNorm[0], uNorm[1], sfr, clumpmap_u_r)

    clumpiness = np.sum(uravel[clumpmap_u_r==3])/np.sum(uravel)

    if showplot:
        return fig, extend_clumpyid+[mextend, fuextend, uextend, vextend], [cc_sm, cc_sfr], clump_agew_r_info
    else:
        fig = 0
        return fig, extend_clumpyid+[mextend, fuextend, uextend, vextend], [cc_sm, cc_sfr], clump_agew_r_info

def _ravel_nonnan_map(clumpmap):
    return clumpmap[~np.isnan(clumpmap)].ravel()

def clumps_contribution(yarray, xarray, densityMap, clumps_map):
    # coords = zip(yarray, xarray)
    # y, x = zip(*coords)
    total = np.sum(10**densityMap[yarray,xarray])

    # import matplotlib.pyplot as plt
    # tmp = np.zeros(densityMap.shape)
    # tmp[yarray,xarray] = 1
    # plt.imshow(densityMap*tmp, origin='lower')
    # plt.show()

    idx = np.where(clumps_map==3)
    if len(idx[0]) == 0:
        clumps_sum = 0.
    elif len(idx[0]) == 1:
        cy = np.array(yarray).ravel()[idx]
        cx = np.array(xarray).ravel()[idx]
        clumps_sum = np.sum(10**densityMap[cy,cx])
    else:
        cy = np.array(yarray).ravel()[idx]
        cx = np.array(xarray).ravel()[idx]

        clumps_coords = zip(cy, cx)
        y, x = zip(*clumps_coords)
        clumps_sum = np.sum(10**densityMap[y,x])

    if clumps_sum == 0.:
        return 1e-99
    else:
        return np.log10(clumps_sum/total)
