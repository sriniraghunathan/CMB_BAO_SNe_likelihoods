import numpy as np, scipy as sc, os, sys, glob, re
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import Gaussian1D, Gaussian2D, GaussianND

sys.path.append('/Users/sraghunathan/Research/SPTpol/analysis/git/CMB_SNIa_3x2pt_Fisher/modules/')
import sne_cmb_fisher_tools, misc

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
                    'nnu': 'neff', 
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
                p1, p2 = params_to_plot[0]
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

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])#,axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    #subax.xaxis.set_tick_params(labelsize=x_labelsize)
    #subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def strip_getdist_latex_str(later_str, what_kind_of_constraint = 'full'):
    if later_str.find('{')>-1:
        delimiters_for_stripping = ['$', '^{+', '}_{', '}$' ]
    else:
        delimiters_for_stripping = ['$', '\\pm', '$']
    regex_for_stripping = '|'.join(map(re.escape, delimiters_for_stripping))
    curr_val_split = re.split(regex_for_stripping, later_str)
    curr_val_split = [c for c in curr_val_split if c.strip()]
    param_label = None
    if what_kind_of_constraint == 'full':
        if later_str.find('=')>-1:
            param_label, curr_val = later_str.split('=')
        elif later_str.find('<')>-1:
            param_label, curr_val = later_str.split('<')
            curr_val = '<%s' %(curr_val)
        elif later_str.find('>')>-1:
            param_label, curr_val = later_str.split('>')
            curr_val = '>%s' %(curr_val)
        #print(param_label, curr_val)
        curr_val = curr_val.strip()
    elif what_kind_of_constraint == 'upper_error':
        curr_val = curr_val_split[1]
    elif what_kind_of_constraint == 'lower_error':
        if len(curr_val_split)==2:
            curr_val = curr_val_split[1]
        elif len(curr_val_split) == 3:
            curr_val = curr_val_split[2]
    elif what_kind_of_constraint == 'best_fit':
        curr_val = curr_val_split[0]

    if what_kind_of_constraint != 'full':
        ##print(curr_val_split, curr_val)
        if curr_val.find('-')>-1:
            curr_val = float( curr_val.strip('-') ) * -1
        else:
            curr_val = float( curr_val )

    return param_label, curr_val

def get_constraints_table(params_to_plot, sample_arr_to_plot, color_arr = None):
    if color_arr is None:
        color_arr = np.tile(None, len( sample_arr_to_plot) )
    #get the constraints
    nx, ny = len(sample_arr_to_plot), len( params_to_plot )
    constraints_dic = {}
    constraints_table = np.empty( (nx, ny), dtype = '<U30' )
    colors_table = np.empty( (nx, ny), dtype = '<U30' )
    col_labels = []
    for pind, ppp in enumerate( params_to_plot ):
        constraints_dic[ppp] = {}
        for sind, (s, c) in enumerate( zip( sample_arr_to_plot, color_arr) ):
            ###print( ppp, s.getLatex(ppp) )
            #tmp = s.getLatex(ppp)
            tmp = s.getInlineLatex(ppp)#, limit = 1, err_sig_figs = 3)
            param_label, curr_val = strip_getdist_latex_str(tmp)
            constraints_table[sind, pind] = r'$%s$' %(curr_val)
            colors_table[sind, pind] = c
            constraints_dic[ppp][sind] = r'$%s$' %(curr_val)
        col_labels.append( r'$%s$' %(param_label) )
    return constraints_dic, constraints_table, colors_table, col_labels

def write_errors_in_diagonal_posteriors(g, params_to_plot, color_arr, constraints_dic, legfsval = 10, legloc = 4, handlelength = 1, handletextpad = 0.3, ncol = 2, frameon = True):
    total_subplots = len( g.subplots )

    if isinstance(legloc, int):
        legloc_arr = np.tile(legloc, total_subplots)
    else:
        legloc_arr = legloc

    for r in range( total_subplots ):
        for c in range( total_subplots ):
            if c!=r: continue
            ax = g.subplots[r,c]
            ppp = params_to_plot[c]
            for sampleind in constraints_dic[ppp]:
                curr_val = constraints_dic[ppp][sampleind]
                ax.plot([], [], color = color_arr[sampleind], label = curr_val)
            
            handles, labels = ax.get_legend_handles_labels()
            leg=ax.legend(handles[sampleind+1:], labels[sampleind+1:], loc = legloc_arr[c], fontsize = legfsval, handlelength = handlelength, handletextpad = handletextpad, ncol = ncol, frameon=frameon)
            leg.get_frame().set_linewidth(0.0)
            #ax.legend(loc = 4, fontsize = legfsval)
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
                     legloc = 4, legcol = 1,  
                     figsize = 4.3, 
                     num_plot_contours = 2, 
                     array_of_samples_to_plot = None, 
                     array_of_params_or_pairs_to_plot = None, 
                     array_of_titles = None, 
                     array_of_labels = None,
                     array_of_leglocs = None, 
                     array_of_legfsval = None,
                     array_of_zoom = None, array_of_sampleinds_for_zoom = None, array_of_param_limits_dic_for_zoom = None,
                     array_of_table_locs = None, array_of_table_width = None, array_of_table_col_width = None,
                     array_to_table_fontsize = None, 
                     array_of_color_arr = None, 
                     array_of_filled_arr = None,
                     array_of_num_plot_contours = None, 
                     array_of_param_limits_dic = None,
                     subplot_size_ratio = 0.95, 
                     scaling = True,
                     cosmo_label = None,
                     show_table = True, 
                     write_errors_on_diagonal = True,
                     diagonal_errors_fsval = 5,
                     diagonal_errors_legloc = 4,
                     **kwargs,
                     ):

    if 'wspace' in kwargs:
        wspace = kwargs['wspace']
    else:
        wspace = 0.12
    if 'yvisibility' in kwargs:
        yvisibility = kwargs['yvisibility']
    else:
        yvisibility = True
    xloc_for_leg, yloc_for_leg = 0.65, 0.85
    if 'xyloc_for_leg' in kwargs:
        if kwargs['xyloc_for_leg'] is not None:
            xloc_for_leg, yloc_for_leg = kwargs['xyloc_for_leg']
    #print(xloc_for_leg, yloc_for_leg ); sys.exit()


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
        if array_of_zoom is None:
            array_of_zoom = np.tile( False, len(samples_to_plot) )

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

        if len( params_or_pairs_to_plot ) == 1:
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

        elif len( params_or_pairs_to_plot ) == 2:
            p1, params_for_pairing_with_p1 = params_or_pairs_to_plot

            g.plots_2d(samples_to_plot, param1 = p1, params2 = params_for_pairing_with_p1, filled=filled, \
                       #lims = [xmin, xmax, ymin, ymax], 
                       legend_labels = '',#labels,
                       colors=color_arr, 
                       nx = len( params_for_pairing_with_p1 )
                       #contour_ls = ls_arr, contour_lw = lw_arr, legend_ncol = len(param_names),s
                       )

            #apply limits and mark axis lines
            for axcntr, curr_ax in enumerate( g.subplots[0] ):
                p1, p2 = p1, params_for_pairing_with_p1[axcntr]
                xmin, xmax = param_limits_dic[p1]
                ymin, ymax = param_limits_dic[p2]
                curr_ax.set_xlim( xmin, xmax )
                curr_ax.set_ylim( ymin, ymax )

                #mark lines
                p1_mod, p2_mod = param_mapping(p1), param_mapping(p2)
                p1val, p2val = param_dict[p1_mod], param_dict[p2_mod]
                lwval, lsval, alphaval, zorderval = 0.5, '-', 0.5, 1
                curr_ax.axvline(p1val, lw = lwval, ls = lsval, alpha = alphaval)#, zorder = zorderval); 
                curr_ax.axhline(p2val, lw = lwval, ls = lsval, alpha = alphaval)#, zorder = zorderval); 

                if p1 == 'mnu':
                    curr_ax.set_xlabel(r'$\sum m_{\nu}$ [eV]')
                elif p2 == 'mnu':
                    curr_ax.set_ylabel(r'$\sum m_{\nu}$ [eV]')

                if p1 == 'H0':
                    curr_ax.set_xlabel(r'$H_{0}$ [km s$^{-1}$ Mpc$^{-1}$]')
                elif p2 == 'H0':
                    curr_ax.set_ylabel(r'$H_{0}$ [km s$^{-1}$ Mpc$^{-1}$]')

            #ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            '''
            legend = g.add_legend(labels, colored_text=False, fontsize = legfsval, legend_loc = legloc, 
                                  handlelength = 1.4, handletextpad = 0.5,
                                  labelspacing = 0.8,
                                 );    
            '''
            fig = plt.gcf()
            print(xloc_for_leg, yloc_for_leg)
            cax = fig.add_axes([xloc_for_leg, yloc_for_leg, 0.25, 0.03], frame_on = True, alpha = 1.)
            for (s, c, l) in zip( samples_to_plot, color_arr, labels):
                if filled:
                    barh(1e10, 1e-10, height = 1e-10, color = c, edgecolor = 'None', label = l)
                else:
                    plot([], [], color = c, label = l)
                ##show(); sys.exit()
            cax.legend(fontsize = legfsval, framealpha = 1., handlelength = 1.4, handletextpad = 0.4, loc = legloc, ncol = legcol, )
            axis('off')            
            
            g.fig.set_figwidth(figsize)
            g.fig.set_figheight(figsize * 4/10.)
            g.fig.subplots_adjust(wspace = -1.)

            
    elif which_plot == 'triangle':
        g.triangle_plot(samples_to_plot, params=params_or_pairs_to_plot, filled=True, \
                        legend_labels = labels, 
                        param_limits = param_limits_dic, \
                        contour_colors=color_arr, 
                        framealpha = 1., 
                        #analysis_settings={'ignore_rows': 0.5},
                        #contour_ls = ls_arr, contour_lw = lw_arr, legend_ncol = len(param_names), param_limits = param_limits_dic, 
                        )

        if write_errors_on_diagonal: #get constraints
            constraints_dic, constraints_table, colors_table, col_labels = get_constraints_table(params_or_pairs_to_plot, samples_to_plot, color_arr)        
            g = write_errors_in_diagonal_posteriors(g, params_or_pairs_to_plot, color_arr, constraints_dic, legfsval = diagonal_errors_fsval, ncol=1, legloc = diagonal_errors_legloc)
        g = mark_axlines(g, params_or_pairs_to_plot, param_dict = param_dict)

        '''
        print(g.subplots)
        leg_ax = g.subplots[1,1]
        print(leg_ax)
        leg = g.add_legend(labels, colored_text=False, fontsize = legfsval, 
                              legend_loc = legloc, 
                              handlelength = 1.4, 
                              handletextpad = 0.4,
                              #labelspacing = 0.8,
                              ax = leg_ax,
                              framealpha = 1.,
                             );
        '''

    elif which_plot == 'multiple_2d_col': #single row

        #define subplots and plot
        lwval = 0.5
        tr, tc = 1, len( array_of_params_or_pairs_to_plot )
        fig = figure(figsize = (10., 4.3))

        '''
        if array_of_param_limits_dic is None:
            subplots_adjust(wspace = 0.05)
        else:
            different_limits = False
            tmp_param_limits_dic = array_of_param_limits_dic[0]
            for tmp in array_of_param_limits_dic:
                if tmp != tmp_param_limits_dic:
                    different_limits = True
            if different_limits:
                subplots_adjust(wspace = 0.12)
            else:
                subplots_adjust(wspace = 0.05)
        '''
        subplots_adjust(wspace = wspace)
        ax_arr = []
        for axcntr in range( tc ):

            #curr samples to plot
            curr_samples_to_plot = array_of_samples_to_plot[axcntr]
            curr_param_pairs_to_plot = array_of_params_or_pairs_to_plot[axcntr]
            curr_colors = array_of_color_arr[axcntr]
            curr_filled = array_of_filled_arr[axcntr]
            curr_labels = array_of_labels[axcntr]
            curr_legloc = array_of_leglocs[axcntr]
            curr_legfsval = array_of_legfsval[axcntr]
            curr_zoom_rect = array_of_zoom[axcntr]
            curr_table_locs = array_of_table_locs[axcntr]
            curr_table_width = array_of_table_width[axcntr]
            curr_table_col_width = array_of_table_col_width[axcntr]
            curr_table_fontsize = array_to_table_fontsize[axcntr]
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
                if curr_param_limits_dic == array_of_param_limits_dic[-1]:
                    param_limits_changed = False
                else:
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
            if p1 == 'H0':
                curr_ax.set_xlabel(r'$H_{0}$ [km s$^{-1}$ Mpc$^{-1}$]')
            elif p2 == 'H0':
                curr_ax.set_ylabel(r'$H_{0}$ [km s$^{-1}$ Mpc$^{-1}$]')

            if axcntr>0:
                if not yvisibility: #not param_limits_changed:
                    setp(curr_ax.get_yticklabels(), visible=False)
                curr_ax.set_ylabel(None)
            else:
                pass


            curr_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            curr_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

            title(r'%s' %(curr_title), fontsize = fsval)
            if cosmo_label is not None and axcntr == 0:
                #xloc, yloc = curr_ax.yaxis.label.get_position()
                lab_fsval = fsval+5
                cosmo_label = r'{\bf Cosmology:} %s' %(cosmo_label)
                if len(cosmo_label)>50:
                    lab_fsval = fsval+2.5
                else:
                    lab_fsval = fsval+7
                #figtext(0.04, 0.5, cosmo_label, fontsize = lab_fsval, rotation = 90., va = 'center')
                figtext(0.03, 0.5, cosmo_label, fontsize = lab_fsval, rotation = 90., va = 'center')

            #g=mark_axlines(g, curr_param_pairs_to_plot, param_dict = param_dict)

            if curr_labels is not None:
                if axcntr == 0:
                    leg = g.add_legend(curr_labels, colored_text=False, fontsize = curr_legfsval, 
                                          legend_loc = curr_legloc, 
                                          handlelength = 1.4, 
                                          handletextpad = 0.4,
                                          #labelspacing = 0.8,
                                          ax = curr_ax,
                                          framealpha = 1.,
                                         );
                else: #legend
                    fig = plt.gcf()
                    cax = fig.add_axes([xloc_for_leg, yloc_for_leg, 0.25, 0.03], frame_on = True, alpha = 1.)
                    for (s, c, l) in zip( curr_samples_to_plot, curr_colors, curr_labels):
                        ##print(axcntr, c, l)
                        if curr_filled:
                            barh(1e10, 1e-10, height = 1e-10, color = c, edgecolor = 'None', label = l)
                        else:
                            plot([], [], color = c, label = l)
                        ##show(); sys.exit()
                    cax.legend(fontsize = curr_legfsval, framealpha = 1., handlelength = 1.4, handletextpad = 0.4, loc = curr_legloc, )
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

            if show_table:

                constraints_dic, constraints_table, colors_table, col_labels = get_constraints_table(curr_param_pairs_to_plot, curr_samples_to_plot, curr_colors)

                tx, ty = curr_table_locs
                tab_width, tab_height = curr_table_width
                table = curr_ax.table(cellText=constraints_table, 
                                colLabels=col_labels, cellLoc = 'center', 
                                bbox = (tx, ty, tab_width, tab_height), 
                                #cellColours = colors_table, 
                                colWidths = curr_table_col_width, 
                                alpha = 1., 
                                zorder = 100.,
                                #edges = 'BT',
                                )
                # Disable auto font size
                table.auto_set_font_size(False)
                table.set_zorder(100)

                # Set the font size
                table.set_fontsize(curr_table_fontsize)

                for cellkey in table.get_celld():
                    table[cellkey].set_linewidth(0.1)
                    if cellkey[0] == 0: continue #header
                    colorval = colors_table[cellkey[0]-1, 0]
                    #print(cellkey, colorval, table[cellkey].get_text())
                    table[cellkey].get_text().set_color(colorval)

            if curr_zoom_rect is not None: #add inset if need be
                curr_param_limits_dic_for_zoom = array_of_param_limits_dic_for_zoom[axcntr]
                xmin, xmax = curr_param_limits_dic_for_zoom[p1]
                ymin, ymax = curr_param_limits_dic_for_zoom[p2]                
                assert array_of_sampleinds_for_zoom is not None
                curr_sampleinds_for_zoom = array_of_sampleinds_for_zoom[axcntr]
                
                curr_samples_to_plot_strip = []
                curr_colors_strip = []
                for iii in curr_sampleinds_for_zoom:
                    curr_samples_to_plot_strip.append( curr_samples_to_plot[iii] )
                    curr_colors_strip.append( curr_colors[iii] )
                
                curr_ax2=add_subplot_axes(curr_ax, curr_zoom_rect)
                g.plot_2d(curr_samples_to_plot_strip, p1, p2, ax = curr_ax2, 
                    filled = curr_filled, 
                    colors = curr_colors_strip,
                    lims = [xmin, xmax, ymin, ymax], 
                    #contours = curr_contours,
                    linewidth = lwval,
                    #labels = curr_labels,
                    add_legend_proxy = True,
                    )
                curr_ax.indicate_inset_zoom(curr_ax2, edgecolor='black', lw = 1., alpha = 0.5)#, ls = '--')
                #ax2.grid(True, which='both', ls='-', lw = 0.1, alpha = 0.1)

                #mark lines
                lwval, lsval, alphaval, zorderval = 0.5, '-', 0.5, 1
                p1_mod, p2_mod = param_mapping(p1), param_mapping(p2)
                p1val, p2val = param_dict[p1_mod], param_dict[p2_mod]
                curr_ax2.axvline(p1val, lw = lwval, ls = lsval, alpha = alphaval)#, zorder = zorderval); 
                curr_ax2.axhline(p2val, lw = lwval, ls = lsval, alpha = alphaval)#, zorder = zorderval);                 

    return g 

def get_param_cov_mat_from_getdist(samples, params):
    cov_mat = samples.cov(params)
    return cov_mat

def make_whisker(samples_to_plot, params_to_plot, param_dict, baseline_sample_ind = 0, labels_arr = None, fsval = 14, barheightwidth = 1, show_table = True, plot_kind = 'ratio', **kwargs):
    if 'color_arr' in kwargs:
        color_arr = kwargs['color_arr']
    else:
        color_arr = [cm.Dark2(int(d)) for d in np.linspace(0, 10, len(samples_to_plot))]
    if 'alphaval' in kwargs:
        alphaval = kwargs['alphaval']
    else:
        alphaval = 1.

    if 'sigma_param_mul_dic' in kwargs:
        sigma_param_mul_dic = kwargs['sigma_param_mul_dic']
    else:
        sigma_param_mul_dic = None

    total_samples = len( samples_to_plot )
    sigma_dic = {}    
    param_labels_dic = {}    
    for cntr in range( total_samples ):
        sigma_dic[cntr] = {}
        curr_samples = samples_to_plot[cntr]
        curr_cov_mat = get_param_cov_mat_from_getdist(curr_samples, params_to_plot)
        
        #constraints
        curr_sigma_arr = np.sqrt( np.diag( curr_cov_mat ) )
        for pind, ppp in enumerate( params_to_plot ):
            sigma_dic[cntr][ppp] = curr_sigma_arr[pind]
            #later_str = curr_samples.getLatex(ppp)
            #param_label = r'$%s$' %( later_str.split('=')[0].strip() )
            #param_label, curr_val = strip_getdist_latex_str(later_str, what_kind_of_constraint = 'full')
            param_label = get_latex_param_str( ppp )
            param_labels_dic[ppp] = param_label

    print( param_labels_dic ) 


    #make whisker plot now
    close('all')
    clf()
    #figure(figsize = (6., 4.2))
    #subplots_adjust(wspace = wspace)
    rowval = 0
    ax = subplot(111)
    for pind, ppp in enumerate( params_to_plot ):
        for cntr in sigma_dic:
            if pind == 0:
                labval = labels_arr[cntr]
            else:
                labval = None
            sigma_val = sigma_dic[cntr][ppp]
            baseline_sigma_val = sigma_dic[baseline_sample_ind][ppp]
            if plot_kind == 'ratio':
                ymin, ymax = 0.7, 1.2
                curr_yval = sigma_val / baseline_sigma_val
                textyloc = ymin-0.04
            elif plot_kind == 'fractional':
                ymin, ymax = 0., 0.22
                curr_yval = (sigma_val - baseline_sigma_val)/baseline_sigma_val
                ###print(ppp, curr_yval)
                textyloc = -0.02

            if cntr != baseline_sample_ind:
                bar(rowval, curr_yval, width = barheightwidth*0.95, color = color_arr[cntr], label = labval, alpha = alphaval)
            if cntr == 2: ##total_samples/2:
                #text(-0.12, rowval-1, param_labels_dic[ppp], fontsize = fsval, ha = 'left')
                if len(param_labels_dic[ppp])>20:
                    text(rowval-2.5, textyloc, r'%s' %(param_labels_dic[ppp]), fontsize = fsval+2, ha = 'left')
                elif len(param_labels_dic[ppp])>15:
                    text(rowval-1., textyloc, r'%s' %(param_labels_dic[ppp]), fontsize = fsval+2, ha = 'left')
                else:
                    text(rowval, textyloc, r'%s' %(param_labels_dic[ppp]), fontsize = fsval+2, ha = 'left')

            if cntr == baseline_sample_ind and not show_table and plot_kind == 'ratio': 
                if sigma_param_mul_dic is not None:
                    mul_fac = sigma_param_mul_dic[ppp]
                else:
                    mul_fac = 1.
                
                sigma_val_to_print = sigma_val * mul_fac
                
                if mul_fac != 1:
                    textval = r'$\sigma$(%s) = %.2f [$10^{%g}$]' %(param_labels_dic[ppp], sigma_val_to_print, np.log10(mul_fac))
                else:
                    textval = r'$\sigma$(%s) = %.2f' %(param_labels_dic[ppp], sigma_val_to_print)
                text(rowval-1.3, 0.6, textval, fontsize = fsval-4, ha = 'left', color = 'white', bbox = dict(facecolor='black', alpha=1., edgecolor='black', boxstyle='round,pad=0.1'))
            rowval +=barheightwidth

        #extra row after each parameter
        rowval+=(0.2 * barheightwidth)

    #ax.set_yticks(list(range(len(params_to_plot))))
    #ax.set_yticklabels([param_labels_dic[ppp] for ppp in params_to_plot], fontsize=fsval)
    ax.tick_params(axis='y', labelsize=fsval)
    ax.set_xticks([])
    #axvspan(0., 1., color = 'silver', alpha = 0.5, zorder = -10)

    if show_table:
        col_labels = [r'{\bf Parameter}', r'{\bf Errors $\sigma_{\rm Fisher}$}']
        parameter_error_table_data = []
        colors_arr = []
        for pind, ppp in enumerate( params_to_plot ):
            baseline_sigma_val = sigma_dic[baseline_sample_ind][ppp]

            if sigma_param_mul_dic is not None:
                mul_fac = sigma_param_mul_dic[ppp]
            else:
                mul_fac = 1.
            
            sigma_val_to_print = baseline_sigma_val * mul_fac
            
            if mul_fac != 1:
                textval = r'%.2f $\times$ $10^{-%g}$' %(sigma_val_to_print, np.log10(mul_fac))
            else:
                textval = r'%.2f' %(sigma_val_to_print)

            parameter_error_table_data.append( [param_labels_dic[ppp], textval] )
            colors_arr.append(['white', 'white'])


        tab_width, tab_height = 0.45, 0.55
        table = ax.table(cellText=parameter_error_table_data, 
                        colLabels=col_labels, cellLoc = 'center', 
                        bbox = (0.03, 0.02, tab_width, tab_height), 
                        cellColours = colors_arr, 
                        colWidths = [0.35, 0.5], alpha = 1., 
                        zorder = 100.,
                        #edges = 'BT',
                        )
        # Disable auto font size
        table.auto_set_font_size(False)

        # Set the font size
        table.set_fontsize(fsval-3)

        for cellkey in table.get_celld():
            table[cellkey].set_linewidth(0.1)
            if cellkey[0] == 0: continue #header
            '''
            colorval = colors_table[cellkey[0]-1, 0]
            #print(cellkey, colorval, table[cellkey].get_text())
            table[cellkey].get_text().set_color(colorval)
            '''

    if plot_kind == 'ratio':
        legloc = 4
        legncol = 1
        ylabval = r'$\sigma / \sigma_{\rm Fisher}$'
    elif plot_kind == 'fractional':
        legloc = 2
        legncol = 3
        ylabval = r'$(\sigma - \sigma_{\rm Fisher}) / \sigma_{\rm Fisher}$'
    ylim(ymin, ymax)
    axhline(1, lw = 1.)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ylabel(ylabval, fontsize = fsval + 6)

    #legend
    legend(loc = legloc, ncol = legncol, fontsize = fsval+1, framealpha = 1., handlelength = 1.4, handletextpad = 0.5)

    grid(True, which = 'both', axis = 'both', lw = 0.1, alpha = 0.05)

    return ax

def get_cosmo_label(cosmo_name):
    cosmo_label_dic = {'lcdm': r'$\Lambda {\rm CDM}$',
                       'mnulcdm': r'$\sum m_{\nu} + \Lambda {\rm CDM}$',
                       'w0walcdm': r'$w_{0} + w_{a} + \Lambda {\rm CDM}$',
                       'nefflcdm': r'$N_{\rm eff} + \Lambda {\rm CDM}$',
                       'neffmnulcdm': r'$N_{\rm eff} + \sum m_{\nu} + \Lambda {\rm CDM}$',
                       'w0wamnulcdm': r'$w_{0} + w_{a} + \sum m_{\nu} + \Lambda {\rm CDM}$',
                      }
    return cosmo_label_dic[cosmo_name]

def get_chain_label(chainname, remove_cmb_datachars = False):
    cmb_exp_dic = {'so_baseline': 'SO-Baseline', 
                   'so_goal': 'SO-Goal', 
                   'advanced_so_baseline': 'ASO-Baseline', 
                   'advanced_so_goal': 'ASO-Goal', 
                   'spt3g': 'SPT-3G', 
                   'spt3gplus': 'SPT-3Gplus',
                   's4_wide': 'CMB-S4',
                  }
    tmpchainname = chainname.replace('-lcdm', '').replace('-w0walcdm', '').replace('-mnulcdm', '').replace('-w0wamnulcdm', '').replace('-w0waomklcdm', '').replace('-omklcdm', '')
    tmpchainname = tmpchainname.replace('-nefflcdm', '').replace('-neffmnulcdm', '')
    dataset_split = tmpchainname.split('+')
    chain_lab = ''
    for ddd in dataset_split:
        ###print(ddd); 
        if ddd == 'lssty3_sne_mock':
            curr_lab = 'LSST-Y3-SNe'
        elif ddd == 'lssty3_sne_mock_binned':
            curr_lab = 'LSST-Y3-SNe (Binned)'
        elif ddd in ['lssty3snesim1_w0walcdm', 'lssty3snesim1_lcdm']:
            curr_lab = 'LSST-Y3-SNe'# (Sim 1)'
        elif ddd == 'desidr2bao_mock':
            curr_lab = 'DESI-DR2-BAO'
        elif ddd in ['desidr3bao_mock', 'desidr3bao_lowz_mock', 'desidr3bao_highz_mock']:
            curr_lab = 'DESI-DR3-BAO'
        elif ddd in ['desy5sne_lcdm', 'desy5sne_w0walcdm']:
            curr_lab = 'DES-Y5-SNe (Data)'
        elif ddd in ['desy5snesim_w0walcdm', 'desy5snesim_lcdm']:
            curr_lab = 'DES-Y5-SNe (Sim)'
        else: #CMB
            cmb_exp_name, cmb_dataset = ddd.split('-')
            curr_lab = '%s-%s' %(cmb_exp_dic[cmb_exp_name], cmb_dataset)
        if remove_cmb_datachars:
            curr_lab = curr_lab.replace('-TTEETEPP', '')
            curr_lab = curr_lab.replace('-TTEETE', '')
        chain_lab = '%s + %s' %(chain_lab, curr_lab)

    chain_lab = chain_lab.strip(' + ')
    
    return chain_lab

def get_cobaya_latex(ppp):
    cobaya_latex_dic = {'logA': '\\log(10^{10} A_\\mathrm{s})', 
                           'As': 'A_\\mathrm{s})', 
                           'ws': 'w_{0}', 'wa': 'w_{a}', 'neff': 'N_\mathrm{eff}', 
                           'mnu': '\sum m_\nu', 
                           'nrun': 'n_\mathrm{run}', 'nrunrun': 'n_\mathrm{run,run}',
                           'theta_MC_100': '100\theta_\mathrm{MC}',
                           'h': 'h', 
                           'H0': 'H_{0}', 
                           'ns': 'n_\\mathrm{s}', 
                           'ombh2': '\\Omega_\\mathrm{b} h^2',
                           'omch2': '\\Omega_\\mathrm{c} h^2', 
                           'tau': '\\tau_\\mathrm{reio}', 
                        }
    latex_val = cobaya_latex_dic[ppp]
    return latex_val

def convert_param_to_latex(param):
    greek_words_small = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 
                        'lambda', 'mu', 'nu', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
    greek_words_captial = [w.capitalize() for w in greek_words_small]
    greek_words = greek_words_small + greek_words_captial
    math_words = ['z']

    tmp_param_split = param.split('_')
    if len( tmp_param_split ) == 1:
        latex_param = r'$%s$' %(param)
    else:
        tmpval = tmp_param_split[0]
        if tmpval in greek_words:
            tmpval = '\%s' %(tmpval)
        latex_param = '%s' %(tmpval)
        braces_arr = ''
        for tmpval in tmp_param_split[1:]:
            if tmpval in greek_words:
                tmpval = '\%s' %(tmpval)
            if tmpval in math_words:
                latex_param = '%s_{%s' %(latex_param, tmpval)
            else:
                latex_param = '%s_{\\rm %s' %(latex_param, tmpval)
            braces_arr = '%s}' %(braces_arr)
        latex_param = '%s%s' %(latex_param, braces_arr)
        latex_param  = r'$%s$' %(latex_param)

    return latex_param

def get_latex_param_str(param):
    params_str_dic= {\
    'norm_YszM': r'${\rm log}(Y_{\ast})$', 'alpha_YszM': r'$\alpha_{_{Y}}$',\
    'beta_YszM': r'$\beta_{_{Y}}$', 'gamma_YszM': r'$\gamma_{_{Y}}$', \
    'alpha': r'$\eta_{\rm v}$', 'sigma_8': r'$\sigma_{\rm 8}$', \
    'one_minus_hse_bias': r'$1-b_{\rm SZ}$', 
    'omega_m': r'$\Omega_{\rm m}$', 'omegam': r'$\Omega_{\rm m}$', \
    'h0':r'$h$', 'm_nu':r'$\sum m_{\nu}$', \
    'ombh2': r'$\Omega_{b}h^{2}$', 'omch2': r'$\Omega_{c}h^{2}$', 'omega_lambda': r'$\Omega_{\Lambda}$',
    'omega_b_h2': r'$\Omega_{b}h^{2}$', 'omega_c_h2': r'$\Omega_{c}h^{2}$',
    'omega_k': r'$\Omega_{k}$',
    'w0': r'$w_{0}$', 'wa': r'$w_{a}$', \
    'tau': r'$\tau_{\rm re}$', 
    'As': r'$A_{\rm s}$', 
    'logA': r'log(10$^{10}$ A$_{s}$)',
    #'As': r'log$A_{\rm s}$', 
    'ns': r'$n_{\rm s}$', 'neff': r'$N_{\rm eff}$', \
    'mnu': r'$\sum m_{\nu}$', 'thetastar': r'$\theta_{\ast}$', \
    'h': r'$h$', 'omk': r'$\Omega_{k}$', 'ws': r'$w_{0}$', \
    'w_0': r'$w_{0}$', 'w_a': r'$w_{a}$', \
    'yhe': r'$Y_{P}$','nnu': r'N$_{\rm eff}$','omegak': r'$\Omega_{k}$',\
    'w': r'$w_{0}$', 'nrun': r'$n_{run}$', 'Aphiphi':r'$A^{\phi\phi}$', \
    'nnu': r'$N_{\rm eff}$', 'H0': r'$H_0$', \
    #adding more
    'a_s': r'$A_{\rm s}$', 'h': r'$h$', 'n_s': r'$n_{\rm s}$', \
    'omega_m': r'$\Omega_{m}$', 
    'omega_b': r'$\Omega_{b}$', 'omegab': r'$\Omega_{b}$',\
    #SNe
    'M': r'$M$', 'alpha': r'$\alpha$', 'beta': r'$\beta$',\
    }

    if param not in params_str_dic:
        return convert_param_to_latex(param)
    else:
        return params_str_dic[param]

def get_gauss_mix_from_fisher(param_dict, f_mat, params, labels, fix_params = ['ws', 'wa', 'mnu', 'neff', 'nrun'], prior_dic = None, get_samples = False): 
    '''
    if cov_mat is None:
        if whichcosmo == 'lcdm':
            fisher_params_latex_label_arr = ['\\log(10^{10} A_\\mathrm{s})', 'H_0' , 'n_\\mathrm{s}', '\\Omega_\\mathrm{c} h^2', '\\Omega_\\mathrm{b} h^2', '\\tau_\\mathrm{reio}']
        #fisher_params_cov_fname = 'data/cmb_data/binned_with_delta_l_100/%s/%s_proposal_covariance_lcdm.txt' %(exp, exp)
        fisher_params_cov_fname = 'data/cmb_data/binned_with_delta_l_100/%s/%s_proposal_covariance_lcdm.txt' %(exp, exp)
        fisher_params_cov = np.loadtxt(fisher_params_cov_fname)
        fisher_params = open( fisher_params_cov_fname, 'r').readlines(1)[0].strip().strip('#').split()
        f_mat = np.linalg.inv( fisher_params_cov )
        f_mat, fisher_params = misc.fix_params(f_mat, fisher_params, fix_params)
        f_mat = misc.add_prior(f_mat, fisher_params, prior_dic)

        cov_mat = np.linalg.inv( f_mat ) 
    '''
    if fix_params is not None:
        if len(fix_params) > 1:
            f_mat, params = misc.fix_params(f_mat, params, fix_params)
    if prior_dic is not None:
        f_mat = misc.add_prior(f_mat, params, prior_dic)

    cov_mat = np.linalg.inv( f_mat ) 
    #print( cov_mat ); sys.exit()
    
    #print( params, np.diag(cov_mat)**0.5 )
    params_mean = []
    for ppp in params:
        if ppp in param_dict:
            params_mean.append( param_dict[ppp] )
        else:
            params_mean.append( 0. )

    ##print(params_mean); sys.exit()

    #print(params_mean)
    gauss_mix = GaussianND(params_mean, cov_mat, names=params, labels = labels)

    if get_samples:
        mix_samples = get_gaussian_mix_from_getdist(gauss_mix)
        return gauss_mix, mix_samples
    else:
        return gauss_mix

def get_gaussian_mix_from_getdist(gauss_mix, total_samples = 50000, label = 'Gaussian mix', random_state = 222, burn_in_fraction = 0.3):
    mix_samples = gauss_mix.MCSamples(total_samples, label=label, random_state = random_state)
    mix_samples.removeBurn(burn_in_fraction)
    return mix_samples

def get_gauss_mix_ori(exp, cov_mat = None, params = None, labels = None, whichcosmo = 'lcdm', fix_params = ['ws', 'wa', 'mnu', 'neff', 'nrun'], prior_dic = None): 
    if cov_mat is None:
        if whichcosmo == 'lcdm':
            fisher_params_latex_label_arr = ['\\log(10^{10} A_\\mathrm{s})', 'H_0' , 'n_\\mathrm{s}', '\\Omega_\\mathrm{c} h^2', '\\Omega_\\mathrm{b} h^2', '\\tau_\\mathrm{reio}']
        #fisher_params_cov_fname = 'data/cmb_data/binned_with_delta_l_100/%s/%s_proposal_covariance_lcdm.txt' %(exp, exp)
        fisher_params_cov_fname = 'data/cmb_data/binned_with_delta_l_100/%s/%s_proposal_covariance_lcdm.txt' %(exp, exp)
        fisher_params_cov = np.loadtxt(fisher_params_cov_fname)
        fisher_params = open( fisher_params_cov_fname, 'r').readlines(1)[0].strip().strip('#').split()
        f_mat = np.linalg.inv( fisher_params_cov )
        f_mat, fisher_params = misc.fix_params(f_mat, fisher_params, fix_params)
        f_mat = misc.add_prior(f_mat, fisher_params, prior_dic)

        cov_mat = np.linalg.inv( f_mat ) 

    if params is None:
        params = fisher_params
        labels = fisher_params_latex_label_arr
    
    #print( params, np.diag(cov_mat)**0.5 )
    params_mean = []
    for ppp in params:
        if ppp in param_dict:
            params_mean.append( param_dict[ppp] )
        else:
            params_mean.append( 0. )

    #print(params_mean)
    return GaussianND(params_mean, cov_mat, names=params, labels = labels)

