"""
Pipeline module 3: find best periods

Search for transit signals in the lightcurves using a BLS algorithm. Creates
an overview figure containing lightcurve, BLS and folded lightcurve for candidate
periods.

Code by Lennart van Sluijs
"""

import time as t
import numpy as np
import os
import matplotlib.pyplot as plt
import operator
from axes_zoom_effect import zoom_effect02
import matplotlib.gridspec as gridspec
import bls # Fotrtan Python implementation

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
#Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 

def fold_data (time, flux, period):
    """
    Fold data for a given period.
    """
    folded = (time % period)
    inds = np.array(folded).argsort()
    time_folded = folded[inds]
    flux_folded = flux[inds]
    
    return time_folded, flux_folded
    
def bin_data(x, nbins, median = False):
    """
    Slices data in nbins, calculates the mean for all bins and returns an array
    containing these, therefore smoothing the data.
    """
    sliced_x = np.array_split(x, nbins)
    
    # replace all value in each bin by median or else the mean
    if median is True:
        binned_x = [np.median(sliced_x[j]) for j in range(0, nbins)]
    else:
        binned_x = [np.mean(sliced_x[j]) for j in range(0, nbins)]
        
    return binned_x
    
def boxval (data, L, H, i1 , i2):
	"""
	Evaluates values of box function at data points.
	
	Input:
	data = data points at which the box function must be evaluated
	L = in-transit flux depth
	H = out-of-transit flux depth
	i1 = indices indicating start of transit
	i2 = indices indicating end of transit
	"""
	box = np.array([L if i1 <= i <= i2 else H for i in range(0,len(data))])
	return box
    
def get_BLS (time, flux, starname, outputfolder, Pmin, Pmax, nP, qmin, qmax, nb):
    """
    Find Box-fitting Least Squares (BLS) frequency spectrum.
    
    Input:
    time = array containing data points of the time series
    flux = array containing data points of the flux series
    Pmin = minimal oribtal period
    Pmax = maximal orbital period
    nP = number of oribtal periods to compute
    qmin = minimal fractional transit length (duration = q * period)
    qmax = maximal fractional transit length
    nb = number of bins in which the data should be binned
    
    Output:
    BLS = BLS frequency spectrum
    P = array containing periods
    best_params = parameters of interest correpsonding to the maximal value in 
    the BLS and two candidate periods ([period, power, fractional transit length, 
    transitdepth, transitcenter])
    """
    
    # calculate the weight for all datapoints
    flux = flux - np.median(flux) # make average signal zero
    #std = np.sqrt( (flux - np.mean(flux))**2 )
    std = np.array([np.std(flux) for i in range(0,len(flux))])
    weight = std**(-2.) * ( sum( 1/std**2 ) )**(-1.)
    
    # find period over which to vary
    #P = np.linspace(Pmin, Pmax, nP)
    P = np.logspace(np.log10(Pmin), np.log10(Pmax), nP)

    # find the signal residue for all periods
    BLS = np.zeros(len(P)) # every period gets assigned one SR forming the BLS
    params_values = np.empty((0,5), int) # stores interesting parameters
    for p in xrange(0, len(P)):
        
        # fold the data
        time_folded, flux_folded = fold_data(time, flux, P[p])
        weight_folded = fold_data(time, weight, P[p])[1]     
        
        # bin the data to fasten code, one must be carefull not to bin the data
        # too much in order resolve the transit signals
        time_folded = bin_data(time_folded, nb, median = False)
        weight_folded = bin_data(weight_folded, nb, median = False)
        flux_folded = bin_data(flux_folded, nb, median = False)
        N = len(time_folded)
        
        power_values = [] # every q gets assigned one D value, see see Kovacs et al. (2002)
        temp = np.empty((0,5), int) # interesting parameters such as transit depth
        # start of transit will be stored here temporarily
        
        for i1 in xrange(0, N-1):
            for i2 in xrange(i1+1, N):
                
                # only if range in [qmin * P, qmax * P] proceed
                q = (time_folded[i2] - time_folded[i1])/P[p]

                if qmin <= q <= qmax:
                    # r,s, power are used to determine the Signal Residue (SR), 
                    # see Kovacs et al. (2002) for more information
                    r = np.sum([ weight_folded[i] for i in range(i1, i2) ])
                    s = np.sum([ weight_folded[i] * flux_folded[i] for i in range(i1, i2) ])
                    power = np.sqrt( (s**2)/(r*(1-r)) ) # sqrt of D without the term that doesn't depend on period
                    power_values = np.append(power_values, power)
                    
                    # ohter interseting parameters
                    depth = power/np.sqrt( r*(1-r) ) # use power as estimae see Kovacs et al. (2002)
                    center = (time_folded[i1] + time_folded[i2])/2
                    params = np.array([P[p], power, q, depth, center])
                    temp = np.vstack((temp, params))
                
                elif q > qmax: # this is done to prevent unnecessary calculations
                    break
                    
        # find values corrseponding with maximal power for given period
        index = np.argmax(power_values)
        BLS[p] = power_values[index] # obtain BLS
        params_values = np.vstack((params_values, temp[index,:]))
        
    # find the indices of best period and two candidate periods
    Pbest, index_Pbest, index_P2, index_P3 = find_candidate_periods(P, BLS)
    
    # parameters corresponding with these periods
    params_Pbest = params_values[index_Pbest]
    params_P2 = params_values[index_P2]
    params_P3 = params_values[index_P3]
    params = np.vstack((params_Pbest, params_P2, params_P3)) # combine into one array
    
    # save as txt file
    data = np.array(zip(P, BLS))
    data_header = '#0 period, 1 SR'
    np.savetxt(os.path.join(outputfolder,'bls_' 
               + str(starname) + '.txt'), data, header=data_header, fmt='%s')
    data = np.copy(params)
    data_header = '#0 best period 1 corresponding SR 2 q 3 transitdepth 4 index at start of transit 5 index at end of transit'
    np.savetxt(os.path.join(outputfolder,'blsparams_' 
               + str(starname) + '.txt'), data, header=data_header, fmt='%s')
               
    # Calculate Signal Detection Efficiency value for best period
    SRbest = params[0,1]
    SDE = (SRbest - np.mean(BLS) )/np.std(BLS)
    
    return P, BLS, params, Pbest, SDE

    
def get_BLS_2 (time, flux, starname, outputfolder, Pmin, Pmax, nP, qmin, qmax, nb):
    """
    Find Box-fitting Least Squares (BLS) frequency spectrum.
    
    Input:
    time = array containing data points of the time series
    flux = array containing data points of the flux series
    Pmin = minimal oribtal period
    Pmax = maximal orbital period
    nP = number of oribtal periods to compute
    qmin = minimal fractional transit length (duration = q * period)
    qmax = maximal fractional transit length
    nb = number of bins in which the data should be binned
    
    Output:
    BLS = BLS frequency spectrum
    P = array containing periods
    best_params = parameters of interest correpsonding to the maximal value in 
    the BLS and two candidate periods ([period, power, fractional transit length, 
    transitdepth, transitcenter])
    """
    
    # calculate the weight for all datapoints
    flux = flux - np.median(flux) # make average signal zero
    P = np.linspace(Pmin,Pmax,nP)
    nf = nP
    fmin = 1./Pmax
    fmax = 1./Pmin
    df = np.diff(P)[0]
    u = np.linspace(fmin,fmin + nf*df,nf)
    v = np.linspace(fmin,fmin + nf*df,nf)
    
    # best period parameters
    results = bls.eebls(time, flux, time, flux, nf, fmin, df, nb, qmin, qmax)
    BLS, best_period, best_power, depth, q, in1, in2 = results
    P = 1./u
    time_f, flux_f = fold_data(time, flux, best_period) # Pbest
    time_f_b = bin_data(time_f, nb, median = False)
    flux_f_b = bin_data(flux_f, nb, median = False)
    center = time_f_b[np.argmin(flux_f_b)]
    params_Pbest = np.array([best_period, best_power, q, depth, center])
    
    # get best three periods and other relevant parameters
    Pbest, P2, P3, best_power_Pbest, best_power_P2, best_power_P3 = find_candidate_periods_2(P, BLS)
    
    time_f, flux_f = fold_data(time, flux, P2) # P2
    time_f_b = bin_data(time_f, nb, median = False)
    flux_f_b = bin_data(flux_f, nb, median = False)
    center = time_f_b[np.argmin(flux_f_b)]
    depth = np.min(flux_f_b)
    params_P2 = np.array([P2, best_power_P2, q, depth, center])
    
    time_f, flux_f = fold_data(time, flux, P3) # P3
    time_f_b = bin_data(time_f, nb, median = False)
    flux_f_b = bin_data(flux_f, nb, median = False)
    center = time_f_b[np.argmin(flux_f_b)]
    depth = np.min(flux_f_b)
    params_P3 = np.array([P3, best_power_P3, q, depth, center])
    
    # combine all the parameters
    params = np.vstack((params_Pbest, params_P2, params_P3))
    
    # save as txt file
    data = np.array(zip(P, BLS))
    data_header = '#0 period, 1 SR'
    np.savetxt(os.path.join(outputfolder,'bls_' 
               + str(starname) + '.txt'), data, header=data_header, fmt='%s')
    data = np.copy(params)
    data_header = '#0 best period 1 corresponding SR 2 q 3 transitdepth 4 index at start of transit 5 index at end of transit'
    np.savetxt(os.path.join(outputfolder,'blsparams_' 
               + str(starname) + '.txt'), data, header=data_header, fmt='%s')
               
    # Calculate Signal Detection Efficiency value for best period
    SRbest = params[0,1]
    SDE = (SRbest - np.mean(BLS) )/np.std(BLS)
    
    return P, BLS, params, Pbest, SDE

def find_candidate_periods(P, BLS, Nmax = 11, perc = 0.02):
    """
    Analyses the BLS spectrum and searches for the two other candidate periods.
    The candidate periods correspond to the second and third highest SR, but under
    condition that the periods are not close to the best period or one of its
    harmonics.
    Radius determines the radius from the best period or one of its harmonics
    in which periods are removed.
    
    Output:
    Indices of best period and indices of two candidate periods
    """

    # find the best period index
    index_Pbest = np.argmax(BLS)
    Pbest = P[index_Pbest]
    
    # calculate harmonics of best period
    #and find best period excluding these regions
    x1 = np.array([1./np.float(n) for n in range(1,Nmax)])
    x2 = np.array([n for n in range(2,Nmax)])
    harmonics = np.sort(np.append(x1, x2)) * Pbest
    
    for n in range(0, len(harmonics)):
        BLS = [0 if harmonics[n]- harmonics[n] * perc <= P[i] <= harmonics[n] + harmonics[n] * perc else
        BLS[i] for i in range(0, len(P)) ] # set values in masked range equal to 0
    index_P2 = np.argmax(BLS)
    P2 = P[index_P2]

    # calculate harmonics of P2 as well
    # and find best period excluding these regions
    harmonics = np.sort(np.append(x1, x2)) * P2
    for n in range(0, len(harmonics)):
        BLS = [0 if harmonics[n]- harmonics[n] * perc <= P[i] <= harmonics[n] + harmonics[n] * perc else
        BLS[i] for i in range(0, len(P)) ] # set values in masked range equal to 0
    index_P3 = np.argmax(BLS)  

    return Pbest, index_Pbest, index_P2, index_P3
    
def find_candidate_periods_2(P, BLS, Nmax = 11, perc = 0.02):
    """
    Analyses the BLS spectrum and searches for the two other candidate periods.
    The candidate periods correspond to the second and third highest SR, but under
    condition that the periods are not close to the best period or one of its
    harmonics.
    Radius determines the radius from the best period or one of its harmonics
    in which periods are removed.
    
    Output:
    Indices of best period and indices of two candidate periods
    """

    # find the best period index
    index_Pbest = np.argmax(BLS)
    best_power_Pbest = np.max(BLS)
    Pbest = P[index_Pbest]
    
    # calculate harmonics of best period
    #and find best period excluding these regions
    x1 = np.array([1./np.float(n) for n in range(1,Nmax)])
    x2 = np.array([n for n in range(2,Nmax)])
    harmonics = np.sort(np.append(x1, x2)) * Pbest
    
    for n in range(0, len(harmonics)):
        BLS = [0 if harmonics[n]- harmonics[n] * perc <= P[i] <= harmonics[n] + harmonics[n] * perc else
        BLS[i] for i in range(0, len(P)) ] # set values in masked range equal to 0
    index_P2 = np.argmax(BLS)
    best_power_P2 = np.max(BLS)
    P2 = P[index_P2]

    # calculate harmonics of P2 as well
    # and find best period excluding these regions
    harmonics = np.sort(np.append(x1, x2)) * P2
    for n in range(0, len(harmonics)):
        BLS = [0 if harmonics[n]- harmonics[n] * perc <= P[i] <= harmonics[n] + harmonics[n] * perc else
        BLS[i] for i in range(0, len(P)) ] # set values in masked range equal to 0
    index_P3 = np.argmax(BLS)
    best_power_P3 = np.max(BLS)
    P3 = P[index_P3]

    return Pbest, P2, P3, best_power_Pbest, best_power_P2, best_power_P3
    
def create_overview_figure_2(time, flux, P, BLS, params, outputfolder, starname, 
                           nb = 200, zoomscale = 1):
    """
    Creates an overview of the systematic error corrected lightcurve,
    the BLS spectrum, the best fitting period and two other candidate periods
    derived from the BLS spectrum. nb should be the same number as bins as used
    during the BLS fitting and zoomscale is an arbitrary scale factor that increases
    the width of the zoomboxes.
    """
    
    # fold data and define important parameters
    Pbest = params[0,0]
    qbest = params[0,2]
    centerbest = params[0,4]
    P2 = params[1,0]
    q2 = params[1,2]
    center2 = params[1,4]
    P3 = params[2,0]
    q3 = params[2,2]
    center3 = params[2,4]
    
    # initialize figure and all subplots
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(5, 6)
    gs.update(wspace = 1, hspace = 1)
    ax1 = plt.subplot(gs[0:3,0:3])
    ax2 = plt.subplot(gs[0:3,3:6])
    ax3 = plt.subplot(gs[3:5,0:2])
    #ax4 = plt.subplot(gs[5:7,0:2])
    ax5 = plt.subplot(gs[3:5,2:4])
    #ax6 = plt.subplot(gs[5:7,2:4])
    ax7 = plt.subplot(gs[3:5,4:6])
    #ax8 = plt.subplot(gs[5:7,4:6])
    
    # overall title
    #plt.suptitle('Star '+ starname, size = 15, y = 0.95)
    
    # fold data for best period
    time_folded, flux_folded = fold_data(time, flux, Pbest)

    # ax1: create lightcurve plot
    ax1.scatter(time, flux, color = 'k', s=0.75)
    ax1.set_xlabel('Time [BKJD]', size = 15)
    ax1.set_ylabel('Normalized flux', size = 15)
    ax1.set_xlim(np.amin(time), np.amax(time))
    ax1.set_ylim(np.amin(flux), np.amax(flux))
    
    #print Pbest*24,P2 * 24,P3 * 24
    # ax2: create BLS spectrum
    ax2.plot(1./P, BLS, color = 'k', lw = 2)
    ax2.set_xlabel('Frequency [1/days]', 
                   size = 15)
    ax2.set_ylabel('SR', size = 15)
    ax2.set_xlim(np.amin(P), np.amax(P))
    ax2.set_ylim(0, np.amax(BLS)*1.2)
    ax2.axvline(x = 1./Pbest, color = tableau20[0], lw = 2, ls = '--')
    ax2.annotate(r'P$_{\rm best} = 5.6 \ \rm{h}$', xy=((1./Pbest-np.amin(1./P))/(np.amax(1./P)-np.amin(1./P)), 0.93), xycoords='axes fraction',
                 color = tableau20[0], size = 15)
    ax2.axvline(x = 1./P2, color = tableau20[6], lw = 2, ls = '--')
    ax2.annotate(r'P$_{\rm 2} = 0.8 \ \rm{h}$', xy=((1./P2-np.amin(1./P))/(np.amax(1./P)-np.amin(1./P)), 0.93), xycoords='axes fraction',
                 color = tableau20[6], size = 15)
    ax2.axvline(x = 1./P3, color = tableau20[4], lw = 2, ls = '--')
    ax2.annotate(r'P$_{\rm 3} = 1.2 \ \rm{h}$', xy=((1./P3-np.amin(1./P))/(np.amax(1./P)-np.amin(1./P)), 0.93), xycoords='axes fraction',
                 color = tableau20[4], size = 15)
    
    """
    # ax3: plot folded lightcurve for best period
    ax3.scatter(time_folded, flux_folded, color = 'k', s=0.75)
    ax3.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax3.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax3.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', 
                    left ='off', right='off', labelleft='off')
    
    """
    time_folded = time_folded * 24.
    # ax4: plot zoom of folded lightcurve for best period
    ax3.scatter(time_folded, flux_folded, color = 'k', s=0.75)
    ax3.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax3.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax3.set_xlabel('Folded time [h]', size = 15)
    ax3.set_ylabel('Normalized flux', size = 15)
    #ax3.annotate(r'P$_{\rm best}$ = ' + str(round(Pbest,4)) + ' d', xy=(0.02, 1.075), xycoords='axes fraction',
    #             color = 'k', size = 12)

    
    # plot smoothend version of lightcurve used to find the BLS
    smoothend_time_folded = bin_data(time_folded, nb, median = False)
    smoothend_flux_folded = bin_data(flux_folded, nb, median = False)
    ax3.plot(smoothend_time_folded, smoothend_flux_folded, color = tableau20[0], lw = '2')
    
    """
    # this parts create a zoom effect
    xmin = centerbest - zoomscale * qbest * Pbest
    xmax = centerbest + zoomscale * qbest * Pbest
    ax3.set_xlim(xmin, xmax)
    zoom_effect02(ax3, ax4)
    """
    
    # fold data again now for P2
    time_folded, flux_folded = fold_data(time, flux, P2)
    time_folded = time_folded * 24. # go to hours
    
    """
    # ax5: plot folded lightcurve for P2 
    ax5.scatter(time_folded, flux_folded, color = 'k',s=0.75)
    ax5.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax5.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax5.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', 
                    left ='off', right='off', labelleft='off')
    """     
    # ax6: plot zoom of folded lightcurve for P2
    ax5.scatter(time_folded, flux_folded, color = 'k',s=0.75)
    ax5.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax5.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax5.set_xlabel('Folded time [h]', size = 15)
    #ax5.set_ylabel('Flux [arbitrary units]', size = 15)
    #ax5.annotate(r'P$_{\rm 2}$ = ' + str(round(P2,4)) + ' d', xy=(0.02, 1.075), xycoords='axes fraction',
    #             color = 'k', size = 15)                    

    # plot smoothend version of lightcurve used to find the BLS
    smoothend_time_folded = bin_data(time_folded, nb, median = False)
    smoothend_flux_folded = bin_data(flux_folded, nb, median = False)
    ax5.plot(smoothend_time_folded, smoothend_flux_folded, color = tableau20[6], lw = '2')
    
    """
    # this parts create a zoom effect
    xmin = center2 - zoomscale * q2 * P2
    xmax = center2 + zoomscale * q2 * P2
    ax5.set_xlim(xmin, xmax)
    zoom_effect02(ax5, ax6)
    """

    # fold data again now for P3
    time_folded, flux_folded = fold_data(time, flux, P3)
    """
    # ax7: plot folded lightcurve for P2 
    ax7.scatter(time_folded, flux_folded, color = 'k',s=0.75)
    ax7.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax7.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax7.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', 
                    left ='off', right='off', labelleft='off')

    """   
    time_folder = time_folded * 24.
    # ax8: plot zoom of folded lightcurve for P3
    ax7.scatter(time_folded, flux_folded, color = 'k',s=0.75)
    ax7.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax7.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax7.set_xlabel('Folded time [h]', size = 15)
    #ax7.set_ylabel('Flux [arbitrary units]', size = 12)
    #ax8.annotate(r'P$_{\rm 3}$ = ' + str(round(P3,4)) + ' d', xy=(0.02, 1.075), xycoords='axes fraction',
    #             color = 'k', size = 15)
                 
    # plot smoothend version of lightcurve used to find the BLS
    smoothend_time_folded = bin_data(time_folded, nb, median = False)
    smoothend_flux_folded = bin_data(flux_folded, nb, median = False)
    ax7.plot(smoothend_time_folded, smoothend_flux_folded, color = tableau20[4], lw = '2')
    """
    # this parts create a zoom effect
    xmin = center3 - zoomscale * q3 * P3
    xmax = center3 + zoomscale * q3 * P3
    ax7.set_xlim(xmin, xmax)
    zoom_effect02(ax7, ax8)
    """
    
    # save figure
    plt.savefig(outputfolder + '/overview_'+ str(starname)+'.pdf')
    plt.savefig(outputfolder + '/overview_'+ str(starname), dpi =300)
    plt.close()
    
def create_overview_figure(time, flux, P, BLS, params, outputfolder, starname, 
                           nb = 200, zoomscale = 1):
    """
    Creates an overview of the systematic error corrected lightcurve,
    the BLS spectrum, the best fitting period and two other candidate periods
    derived from the BLS spectrum. nb should be the same number as bins as used
    during the BLS fitting and zoomscale is an arbitrary scale factor that increases
    the width of the zoomboxes.
    """
    
    # fold data and define important parameters
    Pbest = params[0,0]
    qbest = params[0,2]
    centerbest = params[0,4]
    P2 = params[1,0]
    q2 = params[1,2]
    center2 = params[1,4]
    P3 = params[2,0]
    q3 = params[2,2]
    center3 = params[2,4]
    
    # initialize figure and all subplots
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(7, 6)
    gs.update(wspace = 1, hspace = 1)
    ax1 = plt.subplot(gs[0:3,0:3])
    ax2 = plt.subplot(gs[0:3,3:6])
    ax3 = plt.subplot(gs[3:5,0:2])
    ax4 = plt.subplot(gs[5:7,0:2])
    ax5 = plt.subplot(gs[3:5,2:4])
    ax6 = plt.subplot(gs[5:7,2:4])
    ax7 = plt.subplot(gs[3:5,4:6])
    ax8 = plt.subplot(gs[5:7,4:6])
    
    # overall title
    plt.suptitle('Star '+ starname, size = 15, y = 0.95)
    
    # fold data for best period
    time_folded, flux_folded = fold_data(time, flux, Pbest)

    # ax1: create lightcurve plot
    ax1.scatter(time, flux, color = 'k', s=0.5)
    ax1.set_xlabel('Time [BKJD]', size = 12)
    ax1.set_ylabel('Flux [arbitrary units]', size = 12)
    ax1.set_xlim(np.amin(time), np.amax(time))
    ax1.set_ylim(np.amin(flux), np.amax(flux))
    
    # ax2: create BLS spectrum
    ax2.plot(P, BLS, color = 'k', lw = 2)
    ax2.set_xlabel('Period [days]', 
                   size = 12)
    ax2.set_ylabel('SR', size = 12)
    ax2.set_xlim(np.amin(P), np.amax(P))
    ax2.set_ylim(0, np.amax(BLS)*1.2)
    ax2.axvline(x = Pbest, color = 'r', lw = 2, ls = '--')
    ax2.annotate(r'P$_{\rm best}$', xy=((Pbest-np.amin(P))/(np.amax(P)-np.amin(P)), 0.02), xycoords='axes fraction',
                 color = 'r', size = 12)
    ax2.axvline(x = P2, color = 'g', lw = 2, ls = '--')
    ax2.annotate(r'P$_{\rm 2}$', xy=((P2-np.amin(P))/(np.amax(P)-np.amin(P)), 0.02), xycoords='axes fraction',
                 color = 'g', size = 12)
    ax2.axvline(x = P3, color = 'b', lw = 2, ls = '--')
    ax2.annotate(r'P$_{\rm 3}$', xy=((P3-np.amin(P))/(np.amax(P)-np.amin(P)), 0.02), xycoords='axes fraction',
                 color = 'b', size = 12)
    
    # ax3: plot folded lightcurve for best period
    ax3.scatter(time_folded, flux_folded, color = 'k', s=0.5)
    ax3.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax3.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax3.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', 
                    left ='off', right='off', labelleft='off')
                    
    # ax4: plot zoom of folded lightcurve for best period
    ax4.scatter(time_folded, flux_folded, color = 'k', s=0.5)
    ax4.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax4.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax4.set_xlabel('Folded time [BKJD]', size = 12)
    ax4.set_ylabel('Flux [arbitrary units]', size = 12)
    ax4.annotate(r'P$_{\rm best}$ = ' + str(round(Pbest,4)) + ' d', xy=(0.02, 1.075), xycoords='axes fraction',
                 color = 'k', size = 12)
    
    # plot smoothend version of lightcurve used to find the BLS
    smoothend_time_folded = bin_data(time_folded, nb, median = False)
    smoothend_flux_folded = bin_data(flux_folded, nb, median = False)
    ax4.plot(smoothend_time_folded, smoothend_flux_folded, color = 'r', lw = '2')
    
    # this parts create a zoom effect
    xmin = centerbest - zoomscale * qbest * Pbest
    xmax = centerbest + zoomscale * qbest * Pbest
    ax3.set_xlim(xmin, xmax)
    zoom_effect02(ax3, ax4)
    
    # fold data again now for P2
    time_folded, flux_folded = fold_data(time, flux, P2)
    
    # ax5: plot folded lightcurve for P2 
    ax5.scatter(time_folded, flux_folded, color = 'k',s=0.5)
    ax5.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax5.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax5.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', 
                    left ='off', right='off', labelleft='off')
                 
    # ax6: plot zoom of folded lightcurve for P2
    ax6.scatter(time_folded, flux_folded, color = 'k',s=0.5)
    ax6.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax6.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax6.set_xlabel('Folded time [BKJD]', size = 12)
    ax6.set_ylabel('Flux [arbitrary units]', size = 12)
    ax6.annotate(r'P$_{\rm 2}$ = ' + str(round(P2,4)) + ' d', xy=(0.02, 1.075), xycoords='axes fraction',
                 color = 'k', size = 12)                    
                 
    # plot smoothend version of lightcurve used to find the BLS
    smoothend_time_folded = bin_data(time_folded, nb, median = False)
    smoothend_flux_folded = bin_data(flux_folded, nb, median = False)
    ax6.plot(smoothend_time_folded, smoothend_flux_folded, color = 'g', lw = '2')
    
    # this parts create a zoom effect
    xmin = center2 - zoomscale * q2 * P2
    xmax = center2 + zoomscale * q2 * P2
    ax5.set_xlim(xmin, xmax)
    zoom_effect02(ax5, ax6)

    # fold data again now for P3
    time_folded, flux_folded = fold_data(time, flux, P3)
    
    # ax7: plot folded lightcurve for P2 
    ax7.scatter(time_folded, flux_folded, color = 'k',s=0.5)
    ax7.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax7.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax7.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', 
                    left ='off', right='off', labelleft='off')
                 
    # ax8: plot zoom of folded lightcurve for P3
    ax8.scatter(time_folded, flux_folded, color = 'k',s=0.5)
    ax8.set_xlim(np.amin(time_folded), np.amax(time_folded))
    ax8.set_ylim(np.amin(flux_folded), np.amax(flux_folded))
    ax8.set_xlabel('Folded time [BKJD]', size = 12)
    ax8.set_ylabel('Flux [arbitrary units]', size = 12)
    ax8.annotate(r'P$_{\rm 3}$ = ' + str(round(P3,4)) + ' d', xy=(0.02, 1.075), xycoords='axes fraction',
                 color = 'k', size = 12)
                 
    # plot smoothend version of lightcurve used to find the BLS
    smoothend_time_folded = bin_data(time_folded, nb, median = False)
    smoothend_flux_folded = bin_data(flux_folded, nb, median = False)
    ax8.plot(smoothend_time_folded, smoothend_flux_folded, color = 'b', lw = '2')
    
    # this parts create a zoom effect
    xmin = center3 - zoomscale * q3 * P3
    xmax = center3 + zoomscale * q3 * P3
    ax7.set_xlim(xmin, xmax)
    zoom_effect02(ax7, ax8)
    
    # save figure
    plt.savefig(outputfolder + '/overview_'+ str(starname), dpi =300)
    plt.close()

def find_period(time, flux, outputfolder, starname):
    
    P, BLS, params, Pbest, SDE = get_BLS_2 (time, flux, starname, outputfolder, Pmin=1./24., Pmax=(np.max(time)-np.min(time))/2., 
                      nP= 10000, qmin=1e-4, qmax=0.05, nb = 300)
                      
    create_overview_figure_2(time, flux, P, BLS, params, outputfolder, starname)
    
    return P, BLS, params, SDE
