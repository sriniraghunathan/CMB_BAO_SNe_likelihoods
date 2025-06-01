import numpy as np, scipy as sc, os, sys, glob
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import Gaussian1D, Gaussian2D, GaussianND

##import matplotlib
from pylab import *

#default_param_limits_dic = {'omegam': [0.2, 0.6], 'w': [-1.5, -0.5], 'wa': [-1.8, 0.8]}
default_param_limits_dic = {'omegam': [0.28, 0.35], 
                    'w': [-1.3, -0.5], 
                    'wa': [-1.8, 0.8], 
                    'mnu': [0., 0.24], 
                    'logA': [3.01, 3.1], 
                    'ns': [0.94, 0.99], 
                    'ombh2': [0.022, 0.0227], 
                    'omch2': [0.116, 0.128], 
                    'tau': [0.035, 0.08], 
                    'H0': [65., 69.], 
                   }  
'''
paramfile = 'data/params_cobaya.ini'
param_dict = misc.get_param_dict(paramfile)
param_dict['H0'] = param_dict['h'] * 100.
'''
def param_mapping(p):
    param_map_dic = {'w': 'ws', 
                    'wa': 'wa', 
                    }
    if p in param_map_dic:
        return param_map_dic[p]
    else:
        return p

def mark_axlines(g, params_to_plot, param_dict = None, lwval = 0.5, lsval = '-', alphaval = 0.5, zorderval = 10):
    total_subplots = len( g.subplots )
    for r in range( total_subplots ):
        for c in range( total_subplots ):
            if c>r:continue
            ax = g.subplots[r,c]
            ax.tick_params('both', length=2, width=0.5, which='major', direction = 'in')
            ax.tick_params('both', length=1, width=0.5, which='minor', direction = 'in')

            if total_subplots>1:
                p1, p2 = param_mapping(params_to_plot[c]), param_mapping(params_to_plot[r])
            else:
                p1, p2 = params_to_plot
                p1, p2 = param_mapping(p1), param_mapping(p2)
            ##print(p1, p2)
            if r == c: #diagonal
                if total_subplots>1:
                    ax.yaxis.set_visible(False)
                    if param_dict is not None and p1 in param_dict:
                        pval = param_dict[p1]
                        ax.axvline(pval, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval); 
                else:
                    if param_dict is not None and p1 in param_dict:
                        p1val = param_dict[p1]
                        ax.axvline(p1val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
                    if param_dict is not None and p2 in param_dict:
                        p2val = param_dict[p2]
                        ax.axhline(p2val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
            else:
                if param_dict is not None and p1 in param_dict:
                    p1val = param_dict[p1]
                    ax.axvline(p1val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
                if param_dict is not None and p2 in param_dict:
                    p2val = param_dict[p2]
                    ax.axhline(p2val, lw = lwval, ls = lsval, alpha = alphaval, zorder = zorderval)
    return g

def make_getdist_plot(which_plot, 
                     samples_to_plot, 
                     params_or_pairs_to_plot, 
                     param_dict = None, param_limits_dic = None, 
                     labels = None, 
                     color_arr = ['sandybrown', 'tab:red', 'black', 'darkgreen', 'tab:blue'], 
                     filled = True, 
                     alpha_fill = 0.8, 
                     fsval = 12, legfsval = 12, 
                     legloc = 4, 
                     figsize = 4.3, 
                     num_plot_contours = 2, 
                     array_of_samples_to_plot = None, 
                     array_of_params_or_pairs_to_plot = None, 
                     array_of_titles = None, 
                     array_of_labels = None,
                     array_of_leglocs = None, 
                     array_of_color_arr = None, 
                     array_of_filled_arr = None,
                     array_of_num_plot_contours = None, 
                     array_of_param_limits_dic = None,
                     subplot_size_ratio = 0.95, 
                     scaling = True,
                     ):

    if which_plot.find('multiple_2d')>-1:
        if array_of_samples_to_plot is None:
            array_of_samples_to_plot = [samples_to_plot]
        if array_of_params_or_pairs_to_plot is None:
            array_of_params_or_pairs_to_plot = np.tile( params_or_pairs_to_plot, len(samples_to_plot) )
        if array_of_labels is None:
            array_of_labels = np.tile( labels, len(samples_to_plot) )
        if array_of_color_arr is None:
            array_of_color_arr = np.tile( color_arr, len(samples_to_plot) )
        if array_of_filled_arr is None:
            array_of_filled_arr = np.tile( filled, len(samples_to_plot) )
        if array_of_num_plot_contours is None:
            array_of_num_plot_contours = np.tile( num_plot_contours, len(samples_to_plot) )

    clf()
    if which_plot.find('multiple_2d')>-1:
        g = plots.get_single_plotter(width_inch=figsize)
    else:
        g = plots.get_subplot_plotter(width_inch=figsize)
    g.settings.num_plot_contours = num_plot_contours
    g.settings.subplot_size_ratio = subplot_size_ratio
    g.settings.scaling = scaling
    g.settings.axes_fontsize = fsval
    g.settings.axes_labelsize = fsval + 1
    g.settings.legend_fontsize = legfsval
    g.settings.alpha_filled_add=alpha_fill
    
    if param_limits_dic is None:
        param_limits_dic = default_param_limits_dic

    if which_plot == 'plots_2d':
        #get the limits
        p1, p2 = params_or_pairs_to_plot
        xmin, xmax = param_limits_dic[p1]
        ymin, ymax = param_limits_dic[p2]
        
        g.plots_2d(samples_to_plot, param_pairs=[params_or_pairs_to_plot], filled=filled, \
                   lims = [xmin, xmax, ymin, ymax], 
                   legend_labels = '',#labels,
                   colors=color_arr, 
                   #contour_ls = ls_arr, contour_lw = lw_arr, legend_ncol = len(param_names),s
                   )

        paramnames = params_or_pairs_to_plot
        g = mark_axlines(g, paramnames, param_dict = param_dict)
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        legend = g.add_legend(labels, colored_text=False, fontsize = legfsval, legend_loc = legloc, 
                              handlelength = 1.4, handletextpad = 0.5,
                              labelspacing = 0.8,
                             );        
    elif which_plot == 'triangle':
        g.triangle_plot(samples_to_plot, params=params_or_pairs_to_plot, filled=True, \
                        legend_labels = labels, 
                        param_limits = param_limits_dic, \
                        contour_colors=color_arr, 
                        #analysis_settings={'ignore_rows': 0.5},
                        #contour_ls = ls_arr, contour_lw = lw_arr, legend_ncol = len(param_names), param_limits = param_limits_dic, 
                        )
        g = mark_axlines(g, params_or_pairs_to_plot, param_dict = param_dict)

    elif which_plot == 'multiple_2d_col': #single row

        #define subplots and plot
        lwval = 0.5
        tr, tc = 1, len( array_of_params_or_pairs_to_plot )
        fig = figure(figsize = (10., 4.3))
        if array_of_param_limits_dic is None:
            subplots_adjust(wspace = 0.05)
        else:
            subplots_adjust(wspace = 0.12)
        ax_arr = []
        for axcntr in range( tc ):

            #curr samples to plot
            curr_samples_to_plot = array_of_samples_to_plot[axcntr]
            curr_param_pairs_to_plot = array_of_params_or_pairs_to_plot[axcntr]
            curr_colors = array_of_color_arr[axcntr]
            curr_filled = array_of_filled_arr[axcntr]
            curr_labels = array_of_labels[axcntr]
            curr_legloc = array_of_leglocs[axcntr]
            #curr_contours = array_of_num_plot_contours[axcntr]
            if array_of_titles is not None:
                curr_title = array_of_titles[axcntr]
            else:
                curr_title = None

            if array_of_param_limits_dic is None:
                curr_param_limits_dic = param_limits_dic
                param_limits_changed = False
            else:
                curr_param_limits_dic = array_of_param_limits_dic[axcntr]
                param_limits_changed = True

            #params
            p1, p2 = curr_param_pairs_to_plot
            xmin, xmax = curr_param_limits_dic[p1]
            ymin, ymax = curr_param_limits_dic[p2]

            curr_ax = subplot(tr, tc, axcntr+1)
            g.plot_2d(curr_samples_to_plot, p1, p2, ax = curr_ax, 
                filled = curr_filled, 
                colors = curr_colors,
                lims = [xmin, xmax, ymin, ymax], 
                #contours = curr_contours,
                linewidth = lwval,
                #labels = curr_labels,
                add_legend_proxy = True,
                )

            #mark lines
            lwval, lsval, alphaval, zorderval = 0.5, '-', 0.5, 1
            p1_mod, p2_mod = param_mapping(p1), param_mapping(p2)
            p1val, p2val = param_dict[p1_mod], param_dict[p2_mod]
            curr_ax.axvline(p1val, lw = lwval, ls = lsval, alpha = alphaval)#, zorder = zorderval); 
            curr_ax.axhline(p2val, lw = lwval, ls = lsval, alpha = alphaval)#, zorder = zorderval); 
            if p1 == 'mnu':
                curr_ax.set_xlabel(r'$\sum m_{\nu}$ [eV]')
            elif p2 == 'mnu':
                curr_ax.set_ylabel(r'$\sum m_{\nu}$ [eV]')

            if axcntr>0:
                if not param_limits_changed:
                    setp(curr_ax.get_yticklabels(), visible=False)
                curr_ax.set_ylabel(None)
            else:
                pass


            curr_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            curr_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

            title(r'%s' %(curr_title), fontsize = fsval)

            #g=mark_axlines(g, curr_param_pairs_to_plot, param_dict = param_dict)

            if curr_labels is not None:
                if axcntr == 0:
                    leg = g.add_legend(curr_labels, colored_text=False, fontsize = legfsval, 
                                          legend_loc = curr_legloc, 
                                          handlelength = 1.4, 
                                          handletextpad = 0.4,
                                          #labelspacing = 0.8,
                                          ax = curr_ax,
                                          framealpha = 1.,
                                         );
                else: #legend
                    fig = plt.gcf()
                    xloc, yloc = 0.65, 0.85
                    cax = fig.add_axes([xloc, yloc, 0.25, 0.03], frame_on = True, alpha = 1.)
                    for (s, c, l) in zip( curr_samples_to_plot, curr_colors, curr_labels, curr_filled):
                        ##print(axcntr, c, l)
                        if curr_filled:
                            barh(1e10, 1e-10, height = 1e-10, color = c, edgecolor = 'None', label = l)
                        else:
                            barh(1e10, 1e-10, height = 1e-10, color = None, edgecolor = c, label = l)
                        ##show(); sys.exit()
                    cax.legend(fontsize = legfsval, framealpha = 1., handlelength = 1.4, handletextpad = 0.4, loc = curr_legloc, )
                    axis('off')
                    '''
                    vmin, vmax = 1., max(reqd_delta_z_50_arr)
                    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                    cb = mpl.colorbar.ColorbarBase(cax, cmap=amber_colormap, norm=norm, orientation='horizontal', ticks = amber_tickvals, drawedges = 0., format = r'%g')
                    cb.set_label(r'$\Delta {z_{\rm re, 50}}$', fontsize=fsval-2.5, labelpad = -2.)#-17., position = (0.6,0.38), horizontalalignment='right')
                    cb.ax.tick_params(labelsize=fsval-4.5)
                    cb.minorticks_off()
                    '''

                ###from IPython import embed; embed()
                '''
                import matplotlib
                matplotlib.legend();        
                '''

    return g 


def get_chain_label(chainname):
    cmb_exp_dic = {'so_baseline': 'SO-Baseline', 
                   'so_goal': 'SO-Goal', 
                   'advanced_so_baseline': 'ASO-Baseline', 
                   'advanced_so_goal': 'ASO-Goal', 
                   'spt3g': 'SPT-3G', 
                   's4_wide': 'CMB-S4',
                  }
    tmpchainname = chainname.replace('-lcdm', '').replace('-w0walcdm', '').replace('-mnulcdm', '').replace('-w0wamnulcdm', '')
    dataset_split = tmpchainname.split('+')
    chain_lab = ''
    for ddd in dataset_split:
        ###print(ddd)
        if ddd == 'lssty3_sne_mock':
            curr_lab = 'LSST-Y3-SNe'
        elif ddd == 'desidr2bao_mock':
            curr_lab = 'DESI-DR2-BAO'
        elif ddd == 'desy5sne_w0walcdm':
            curr_lab = 'DES-Y5-SNe (Data)'
        elif ddd == 'desy5snesim_w0walcdm':
            curr_lab = 'DES-Y5-SNe (Mock)'
        else: #CMB
            cmb_exp_name, cmb_dataset = ddd.split('-')
            curr_lab = '%s-%s' %(cmb_exp_dic[cmb_exp_name], cmb_dataset)
        chain_lab = '%s+%s' %(chain_lab, curr_lab)
    chain_lab = chain_lab.strip('+')
    
    return chain_lab