"""
Pipeline module 2: correct for motion of the satelite

The satlite suffers from unstable poynting. This must be corrected in order to
be able to analyze the data.

Code by Lennart van Sluijs
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pyfits
from scipy.ndimage import measurements
import scipy.interpolate
from scipy import signal
import numpy.ma as ma
from sklearn.decomposition import PCA
from numpy.polynomial import polynomial as P
from scipy import interpolate
import operator
from mpl_toolkits.axes_grid1 import make_axes_locatable

def remove_NaN_zero_inf_neg_values(time, flux):
    """
    Removes NaN values from your array.
    """
    
    # remove NaN values
    time = np.delete(time, np.where(np.isnan(flux)))
    flux = np.delete(flux, np.where(np.isnan(flux)))

    # remove zero and negative values
    time = np.delete(time, np.where(flux <= 0))
    flux = np.delete(flux, np.where(flux <= 0))
    
    # remove infinte values
    time = np.delete(time, np.where(np.isinf(flux)))
    flux = np.delete(flux, np.where(np.isinf(flux)))
    
    return time, flux
    
def remove_inf_Xcof_Ycof(time, flux, Xcof, Ycof):
    """
    Removes invalid Xcof and Ycof values.
    """

    # remove infinte values
    time = np.delete(time, np.where(np.isinf(Xcof)))
    flux = np.delete(flux, np.where(np.isinf(Xcof)))
    Ycof = np.delete(Ycof, np.where(np.isinf(Xcof)))
    Xcof = np.delete(Xcof, np.where(np.isinf(Xcof)))

    return time, flux, Xcof, Ycof
    
def bin_data(x, nbins, median = False):
    """
    Slices data in nbins, calculates the mean for all bins and returns an array
    containing these, therefore smoothing the data.
    """
    sliced_x = np.array_split(x, nbins)
    
    # replace all value in each bin by median or else the mean
    if median is True:
        binned_x = [np.nanmedian(sliced_x[j]) for j in range(0, nbins)]
    else:
        binned_x = [np.nanmean(sliced_x[j]) for j in range(0, nbins)]
        
    return binned_x
    
def get_centerofflux(pixflux, time, aperture, outputfolder, starname, plot = False):
    """
    Find centroid values of X and Y contained withing the aperture.
    """
    
    # find center of mass
    pixfluxinaperture = np.nan_to_num(pixflux) #* aperture #flux in aperture and deal with nan values
    Ycof = np.array([measurements.center_of_mass(pixfluxinaperture[i])[0] for i in range(0,len(pixflux)) ])
    Xcof = np.array([measurements.center_of_mass(pixfluxinaperture[i])[1] for i in range(0,len(pixflux)) ])
    
    # get centroid relative to the center of the aperture
    Yc_aper, Xc_aper = measurements.center_of_mass(aperture) #define center of aperture
    Ycof = Ycof - Yc_aper
    Xcof = Xcof - Xc_aper
    
    # plot found values
    if plot is True:
        plt.figure()
        ax = plt.gca()
        #plt.title('Center of flux', size = 15)
        im = plt.scatter(Xcof, Ycof, c=time, cmap = 'viridis', s=10, marker='o',edgecolor='none')
        plt.xlabel('X', size = 15)
        plt.ylabel('Y', size = 15)
        #plt.axis('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        cax.set_ylabel('Time [BKJD]', size = 15)
        plt.tight_layout()
        #cbar.set_label('Time (BKJD)', rotation=270, size = 12)
        # time should be provided if one wants to plot
        plt.savefig(outputfolder + '/cof_'+ str(starname)+'.pdf')
        plt.savefig(outputfolder + '/cof_'+ str(starname), dpi =300)
        plt.close()
        
    return Xcof, Ycof
    
def remove_thrusteronpoints(time, flux, Xcof, Ycof, outputfolder, starname, cutoff = 3, no_iter = 3, dr=1, plot = True):
    """
    Removes those points on which the thrusters where likely fired. If the distance
    between two succeeding centroids changes more then cutoff * median(centroids)
    this is identified as a thruster event and the data point before and after 
    this thruster event are removed.         
    """
    
    # initialize plot if plot is True
    if plot is True:
        plt.scatter(time, flux, color = 'r', label = 'removed', s = 0.5)

    for i in range(no_iter):
        
        # identify thruster events and mask outliers
        dist_cof = np.sqrt( np.diff(Xcof)**2 + np.diff(Ycof)**2 )
        indices = np.array(np.where(abs(dist_cof - np.mean(dist_cof)) > cutoff * np.std(dist_cof)))-np.int(dr/2)
        indices = np.delete(indices, np.where(indices<0))
        
        # remove closest points to thruster event as well
        for j in range(dr):
            indices = np.concatenate((indices,indices+1))
        indices = np.unique(indices)
                 
        # remove thruster events
        flux = np.delete(flux, indices)
        time = np.delete(time, indices)
        Xcof = np.delete(Xcof, indices)
        Ycof = np.delete(Ycof, indices)
    
    # plot which data point are removed during this procedure
    if plot is True:
        plt.title('Thruster event removal', size = 15)
        plt.scatter(time, flux, color = 'k', label = 'kept', s = 0.5)
        plt.legend()
        plt.xlabel('Time [BKJD]', size = 12)
        plt.ylabel('Flux (arbitary units)', size = 12)
        plt.savefig(outputfolder + '/thrusterevents_'+ str(starname), dpi =300)
        #plt.show()
        plt.close()
    
    return time, flux, Xcof, Ycof
    
def remove_outliers(time, flux, Xcof, Ycof, outputfolder, starname, cutoff = 2, plot = True):
    """
    Removes outliers who lie very far away from a polynominal fit to whole flux.
    Data for which flux - fit < cutoff * sigma are figured to be outliers and are removed.
    This means only outliers above the mean are removed. Other outliers could be transits.
    """
    
    # fit a polynominal
    coef = P.polyfit(time, flux, 4)
    polyfit = P.polyval(time, coef)
    
    # remove cutoff * sigma outliers where here sigma = RMS
    sigma = np.sqrt( sum( (flux - polyfit)**2 ) / len(flux) )
    poyntingjitter_mask = flux - polyfit < cutoff * sigma

    # plot which data point are removed during this procedure
    if plot is True:
        plt.title('Outlier removal', size = 15)
        plt.plot(time, polyfit, color = 'g', label = 'best polynominal fit', lw = 1)
        #plt.plot(time, flux, color = 'k')
        plt.scatter(time[:][np.invert(poyntingjitter_mask)], flux[:][np.invert(poyntingjitter_mask)],
                    color = 'r', label = 'removed data', s=0.5)
        plt.scatter(time[:][poyntingjitter_mask], flux[:][poyntingjitter_mask], color = 'k', label = 'kept data', s=0.5)
        plt.legend()
        plt.xlabel('Time [BKJD]', size = 12)
        plt.ylabel('Flux (arbitary units)', size = 12)
        plt.savefig(outputfolder + '/outlierremoval_'+ str(starname), dpi =300)
        plt.close()
    
    # remove those data points
    time = time[:][poyntingjitter_mask]
    flux = flux[:][poyntingjitter_mask]
    Xcof = Xcof[:][poyntingjitter_mask]
    Ycof = Ycof[:][poyntingjitter_mask]
    polyfit = polyfit[:][poyntingjitter_mask]
    
    return time, flux, Xcof, Ycof
    
def remove_outliers_for_fit(x, y, nbins = 5, cutoff = 2.5):
    """
    Bins data in nbins bins and than calculates the mean and std for each bin.
    If | X_i - mean | > cutoff * std_bin these point are removed.
    This is done such that the fitting procedure will be less affected by outliers.
    """
    
    # sort arrays for increasing x value
    L = sorted(zip(x,y), key=operator.itemgetter(0))
    x, y = zip(*L)
    
    # slice the Xcof values again in nbins bins
    sliced_x = np.array_split(x, nbins)
    sliced_y = np.array_split(y, nbins)
    binsize = [len(sliced_x[j]) for j in range(0,nbins)]    
    mean = [np.mean(sliced_y[j]) for j in range(0, nbins)]
    std = [np.std(sliced_y[j]) for j in range(0, nbins)]
    
    # slice data in different parts and calculate mean and std for all parts
    mean = [mean[j] for j in range(0,nbins) for k in range(0,binsize[j])]
    std = [std[j] for j in range(0,nbins) for k in range(0,binsize[j])]
    mask = [ abs(y[j] - mean[j]) < cutoff * std[j] 
             for j in range(0, len(x))]
                 
    # mask the data
    masked_x = [x[j] for j in range(0, len(x)) if mask[j]]
    masked_y = [y[j] for j in range(0, len(x)) if mask[j]]
    
    return masked_x, masked_y
    
def remove_poyntingjitter(time, flux, Xcof, Ycof, outputfolder, starname, noslices = 11, 
                          noiterations = 3, nbins = 11, plot = False, 
                          plot_intermediate_step = False, delete_firstpart = False,
                          delete_gap = False, median_filter = False, gap = 2016,
                          delete_lastpart = False, robust = True):
    """
    Removes the flux changes due to motion of the satelite.
    noslices is number of slices and should be an even number.
    noiterations is number of times the fitting procedure will be repeated to improve the fit.
    """
    for n in range(noiterations):
        # decorrelate with time
        mask = [i for i in range(len(time))]
        for i in range(3):
            L = int(np.max(time)-np.min(time)) # bin for approximately 24 hours
            if L >= len(time): L = int(len(time)/2.) # make sure L is not too large
            time_binned = bin_data(time[mask], nbins = L, median = True)
            flux_binned = bin_data(flux[mask], nbins = L, median = True)
            tck = scipy.interpolate.splrep(time_binned, flux_binned)
            longterm_trend = scipy.interpolate.splev(time, tck)
            mask = [i for i in range(len(flux)) if abs(flux[i] - longterm_trend[i]) < 3 * np.std(flux) ]
        flux = flux/longterm_trend
        
        # slice data with interger number of thruster rolls
        dist_cof = np.sqrt( np.diff(Xcof)**2 + np.diff(Ycof)**2 )
        indices = abs(dist_cof - np.mean(dist_cof)) < 3 * np.std(dist_cof)
        indices = np.array([ int(len(indices)/noslices * i) for i in range(1, noslices) ])
        
        # slice data
        sliced_flux = np.array_split(flux, indices) # this slices flux, not always exactly the same slice size is obtained
        sliced_time = np.array_split(time, indices)
        sliced_Xcof = np.array_split(Xcof, indices)
        sliced_Ycof = np.array_split(Ycof, indices)
        
        for i in range(noslices):
            # find the directions of the eigenvectors describing the data in the best using PCA
            cof= np.array(zip(sliced_Xcof[i], sliced_Ycof[i])) # combine the Xcof and Ycof values in one array
            pca = PCA(n_components = 2) # we want to perform PCA in 2 dimensions
            pca.fit(cof) # perform ftting procedure
            v1, v2 = pca.components_ # eigenvectors found
        
            # rotate all vallues to new coordinate frame
            theta = -1*np.arctan(v1[1]/v1[0])
            rotmatrix = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
            cof = np.array([np.dot(rotmatrix, cof[j,:]) for j in range(0,len(cof))]) # center of flux
            sliced_Xcof[i] = cof[:,0] # new rotated coordinated x' and y'
            
            # mask some outliers during the fitting
            masked_sliced_Xcof, masked_sliced_flux = remove_outliers_for_fit(sliced_Xcof[i], sliced_flux[i], nbins = 5, cutoff = 2)
            if len(masked_sliced_Xcof) is 0: masked_sliced_Xcof = sliced_Xcof[i] # to make sure not all data is removed
            if len(masked_sliced_flux) is 0: masked_sliced_flux = sliced_flux[i]
            # find dependence of the flux on x'
            coef = P.polyfit(masked_sliced_Xcof, masked_sliced_flux, 4)
            polyfit_flux = P.polyval(sliced_Xcof[i], coef, 4) # F(y') flux dependence on y coordinate centroid
                        
            # plot all intermediate steps if true
            if plot_intermediate_step is True:
                if [i,n] == [3,0]:
                    #plt.title('Correlation flux and centroid', size = 15)        
                    plt.scatter(sliced_Xcof[i], sliced_flux[i], color = 'r', label = 'excluded')
                    plt.scatter(masked_sliced_Xcof, masked_sliced_flux, color = 'k', label = 'data')
                    xmin = np.amin(sliced_Xcof[i]) # define interval to plot best fit
                    xmax = np.amax(sliced_Xcof[i])
                    interval = np.linspace(xmin, xmax, 100)
                    plt.plot(interval, P.polyval(interval, coef), color = 'g', label = 'best polynominal fit', lw = 2)
                    plt.legend(loc = 1, fontsize = 15,scatterpoints=1)            
                    plt.xlabel('Deprojected x-coordinate of center of flux', size = 15)
                    plt.ylabel('Flux (arbitrary units)', size = 15)
                    plt.savefig(outputfolder + '/slice_' + str(i)+str(n) + '_fluxcentroid_'+ str(starname)+'.pdf')
                    plt.savefig(outputfolder + '/slice_' + str(i)+str(n) + '_fluxcentroid_'+ str(starname), dpi =300)
                    #plt.show()
                    plt.close()
                    
            # decorelate with centroid motion   
            sliced_flux[i] = sliced_flux[i] / polyfit_flux
        
        # combine again all pieces of the flux to a full lightcurve
        time = np.concatenate(sliced_time)
        flux = np.concatenate(sliced_flux)
    
    # apply median filter
    if median_filter is True:
        time, flux = apply_median_filter(time, flux, outputfolder, starname, plot = True)
            
    if robust is True:
        # account for some regions where fitting might have been going horribly wrong
        # dont use this one in combination with one of the others
        std_sliced = [np.std(sliced_flux[i]) for i in range(noslices) ]
        std = np.std(flux)
        time_org = np.copy(time)
        for i in range(noslices):
            if std_sliced[i] > 2 * std:
                i_min = i-1 if i-1 >= 0 else 0
                t_min_robust = time_org[indices[i_min]]
                t_max_robust = time_org[indices[i]] if i < len(indices) else time_org[-1]
                time, flux, Xcof, Ycof = remove_time_range(time, flux, Xcof, Ycof, t_min = t_min_robust, t_max = t_max_robust)
                #N_max += -1
    return time, flux, Xcof, Ycof

def apply_median_filter(time, flux, outputfolder, starname, kernel_size= 101, plot = True):
    """
    Perform a medium filter to remove other trends. Divide data by the median.
    kernel_size should be odd.
    """
    
    # perform median filter
    flux = flux/signal.medfilt(flux, kernel_size)
    
    # plot if plot is True
    if plot is True:
        plt.title('Median filtered lightcurve', size = 15)
        plt.scatter(time, flux, color = 'k')
        # time should be provided if one wants to plot
        plt.xlabel('Time [BKJD]', size = 12)
        plt.ylabel('Flux [arbitrary units]', size = 12)
        #plt.ylim(0.997,1.003)
        plt.savefig(outputfolder + '/medianfilteredlc_'+ str(starname), dpi =300)
        plt.close()
           
    return time, flux
    
    
def flux_modulation_model(time, polyfit_longterm, polyfit_shortterm, outputfolder, starname, plot = True):
    """
    Reconstructs the total modulation of the flux due to the satelite motion.
    """
    # combine shortterm and longterm modulations
    model = polyfit_longterm * polyfit_shortterm
        
    # plot if plot is True
    if plot is True:
        plt.title('Model of flux modulation due to satelite motion', size = 15)
        plt.scatter(time, model, color = 'g')
        plt.xlabel('Time [BKJD]', size = 12)
        plt.ylabel('Flux [arbitrary units]', size = 12)
        plt.savefig(outputfolder + '/fluxmodulation_'+ str(starname), dpi =300)
        plt.close()
        
    return model

def remove_time_range(time, flux, Xcof, Ycof, t_min, t_max):
    """
    Remove all data points in a certain time range (t_min, t_max), which is 
    known to contain bad flux values.
    """
    
    # get indices in time range (t_min, t_max)
    indices = [i for i in range(0, len(time)) if t_min <= time[i] <= t_max]
    
    # remove these points
    time = np.delete(time, indices)
    flux = np.delete(flux, indices)    
    Xcof = np.delete(Xcof, indices)
    Ycof = np.delete(Ycof, indices) 
    
    return time, flux, Xcof, Ycof
    
def correct_for_satelite_motion(time, pixflux, flux, aperture, outputfolder, starname):
    """
    Correct for systematic errors due to motion of the satelite.
    """
    
    Xcof, Ycof = get_centerofflux(pixflux, time, aperture, outputfolder, starname, plot = True)

    time, flux, Xcof, Ycof = remove_inf_Xcof_Ycof(time, flux, Xcof, Ycof)

    time, flux, Xcof, Ycof = remove_thrusteronpoints(time, flux, Xcof, Ycof, outputfolder, starname, plot = True)
    
    time, flux, Xcof, Ycof = remove_poyntingjitter(time, flux, Xcof, Ycof, outputfolder, starname, plot = True)
    
    time, flux, Xcof, Ycof = remove_outliers(time, flux, Xcof, Ycof, outputfolder, starname, plot = True)
    
    time, flux = remove_NaN_zero_inf_neg_values(time, flux)
    
    # save as txt file
    data = np.array(zip(time, flux))
    data_header = '#0 time, 1 flux'
    np.savetxt(os.path.join(outputfolder,'systematiccorrectedlc_' 
               + str(starname) + '.txt'), data, header=data_header, fmt='%s')

    # create plot of corrected lightcurve
    plt.title('Systematic corrected lightcurve', size = 15)
    plt.scatter(time, flux, color = 'k', s=0.5)
    # time should be provided if one wants to plot
    plt.xlabel('Time [BKJD]', size = 12)
    plt.ylabel('Flux [arbitrary units]', size = 12)
    #plt.ylim(0.997,1.003)
    #plt.show()
    plt.savefig(outputfolder + '/systematiccorrectedlc_'+ str(starname), dpi =300)
    plt.close()
    
    return time, flux
