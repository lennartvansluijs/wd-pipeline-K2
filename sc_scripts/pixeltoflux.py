"""
Pipeline module 1: perform circular aperture photometry

Go from MAST pixel target files to a raw flux.

Code by Lennart van Sluijs
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pyfits
from scipy.ndimage import measurements
from matplotlib.colors import LogNorm
from correctformotion import *

def absdiff2d(A,B):
    """
    Absolute difference between two arrays 2d arrays A and B.
    Returns unique and not common elements as another 2d array C.
    """
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    cset = aset-bset | bset-aset
    C = np.array([x for x in cset])
    return C
    
def join2d(A,B):
    """
    Absolute difference between two arrays 2d arrays A and B.
    Returns unique and not common elements as another 2d array C.
    """
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    cset = aset | bset
    C = np.array([x for x in cset])
    return C

def diff2d(A,B):
    """
    Absolute difference between two arrays 2d arrays A and B.
    Returns unique and not common elements as another 2d array C.
    """
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    cset = aset-bset
    C = np.array([x for x in cset])
    return C

def arg_local_minima(x):
    """
    INPUT:
    x - 1d array
    
    OUTPUT:
    indices - indices of local minima, empty if none exist
    """
    diff = np.diff(x)
    indices = [i+1 for i in range(len(diff)-1) if diff[i]<0 and diff[i+1]>0]
    #indices = [i for i in range(1, len(diff)-1) if diff[i]<diff[i-1] & diff[i]<diff[i+1]]
    return indices

def get_pixelpixflux (starname, filename, inputfolder='',outputfolder='', remove = True):
    """
    Reads a pixel file and returns time and pixel pixflux over time.
    If remove is True, images with quality tag other than 0 are removed, because
    the quality of those points is known to be bad.
    """
    
    # open pixel FITS file
    pixelfile = pyfits.open(inputfolder+filename, memmap = True)
    L = len(pixelfile[1].data) # no. images\
    
    # extract relevant imformation
    time = np.array([pixelfile[1].data[i]['TIME'] for i in range(0,L)])
    pixflux = np.array([pixelfile[1].data[i]['FLUX'] for i in range(0,L)])
    quality = np.array([pixelfile[1].data[i]['QUALITY'] for i in range(0,L)])
    
    keplerid = pixelfile[0].header['KEPLERID']
    kepmag = pixelfile[0].header['Kepmag']
    obsmode = pixelfile[0].header['OBSMODE']
    channel = pixelfile[0].header['CHANNEL']
    module = pixelfile[0].header['MODULE']
    RA = pixelfile[0].header['RA_OBJ']
    DEC = pixelfile[0].header['DEC_OBJ']
    
    # get X and Y position of the star in the FITS file
    X_s = pixelfile[2].header['CRPIX1']
    Y_s = pixelfile[2].header['CRPIX2']
    
    info = [keplerid, kepmag, obsmode, channel, module, RA, DEC]
    info_head = '# 0 kepid, 1 kepmag, 2 obsmode, 3 channel, 4 module, 5 RA, 6 DEC'
    np.savetxt(os.path.join(outputfolder,'info_' + str(starname) + '.txt'), info, header=info_head, fmt='%s')
    
    # remove data known to have a bad quality inmediately
    if remove is True:
        flags = np.array([1,2,4,8,16,32,64,128,256,1024,40960,8192,16384,32768,65536,12288,98304,131072,262144,524288,1089568, 1081376,1056768,1048576,1081344],dtype='int')# flags to be removed
        for i in range(0,len(flags)):
            time = np.delete(time, np.where(quality == flags[i]))
            pixflux = np.delete(pixflux, np.where(quality == flags[i]), axis = 0)
            quality = np.delete(quality, np.where(quality == flags[i]), axis = 0)
            #time = np.delete(time, np.where(quality != 0))
            #pixflux = np.delete(pixflux, np.where(quality != 0), axis = 0)

    # get campaign
    campaign = pixelfile[0].header['CAMPAIGN']

    return time, pixflux, kepmag, X_s, Y_s, campaign

def get_threshold(flux, magnitude, cutoff = 3):
    """
    Calculates the threshold. Pixels above the threshold are thought to belong
    to stellar objects.
    """
    
    # estimate the background of the objects
    background = np.median([flux.ravel()[i] for i in range(len(flux.ravel())) 
    if abs(flux.ravel()[i]-np.mean(flux.ravel())) < np.std(flux.ravel()) * cutoff])
    
    # caculate percentage above background the star need to have
    # based on the magnitude of the star following Sanchis-Ojeda et al. (2015)
    # DISCOVERY OF THE DISINTEGRATING ROCKY PLANET K2-22b WITH A
    # COMETARY HEAD AND LEADING TAIL
    if magnitude < 11.5:
        perc = 0.3 # 30%
    elif 11.5 <= magnitude <= 14:
        # do a linear interpolation for magnitudes in between
        a = (0.04-0.3)/(14-11.5)
        b = 0.3 - 11.5 * a
        perc = a * magnitude + b
    else:
        perc = 0.04 # 4%
    threshold = background * (1 + perc)

    return threshold

"""
def get_aperture (pixflux, magnitude, outputfolder, starname, cutoff = 3, plot = True):

    Find the aperture over which the total flux of the star will
    be determined.
    Here detecionlvl defines the cutoff flux used to identify objects
    in the image where the cutoff flux = cutoff * median(flux).

    # add pixflux over complete data set
    flux = np.nansum(pixflux, axis = 0)
    
    # this part identifies objects and removes the smallest
    threshold = get_threshold(flux, magnitude)
    segmentationimage = np.array(flux > threshold)
    lw, num = measurements.label(segmentationimage) # label objects
    areas = np.bincount(lw.ravel(),segmentationimage.ravel()) # determine areas
    aperture = (lw == np.argmax(areas))*1 # pick only the largest
    
    # define center of mass of the aperture
    Ycof, Xcof = measurements.center_of_mass(flux * segmentationimage)
    
    # this part smoothens aperture to a circular aperture
    aperradius = np.sqrt(np.max(areas)/np.pi)
    y, x = np.indices((aperture.shape))
    r = np.sqrt((x-Xcof)**2 + (y-Ycof)**2)    
    #aperture = (r < aperradius)*1

    outline = make_aperture_outline(aperture) # a new outline (ONLY for figure)
    # create a plot of combined flux and aperture
    if plot is True:
        plt.imshow(flux, cmap = 'Reds_r', interpolation = 'none')
        plt.colorbar()
        plt.plot(outline[:, 0], outline[:, 1],color='b', zorder=10, lw=2.5)#,label=str(kepmag))
        plt.xlabel('X', size = 12)
        plt.ylabel('Y', size = 12)
        plt.xlim(0,aperture.shape[1])
        plt.ylim(0,aperture.shape[0])
        plt.savefig(outputfolder + '/aperture_'+ str(starname), dpi =300)
        plt.close()
        
    return aperture, aperradius
"""

def get_aperture(inputfolder, outputfolder, filename, starname, pixel_lim = 25, plot = True, intermediate_plot = False):
    """
    Find the aperture over which the total flux of the star will
    be determined.
    This will be done by trying different apertures around the star center
    and choose the one which has the smallest noise in the reduced lightcurve.
    
    OUTPUT PARAMETERS:
    best_aperture - the best aperture for this star
    """
    
    # get raw data
    raw_time, raw_pixflux, magnitude, X_s, Y_s = get_pixelpixflux(starname, filename, inputfolder, outputfolder, remove = True)

    # get added fluc values and array containing the indices
    raw_flux = np.nansum(raw_pixflux, axis = 0)

    # get dimensions and indices array of rawflux
    Nx, Ny = raw_flux.shape
    Nx += 1
    Ny += 1
    img_indices = np.array([[i,j] for i in range(raw_flux.shape[0]) for j in range(raw_flux.shape[1])]).reshape((raw_flux.shape[0],raw_flux.shape[1],2))
    
    # use FITS file position as star center estimate
    Y_s = Y_s if Y_s >=1 else 1 # make sure pixel values are inside the image
    X_s = X_s if X_s >=1 else 1
    Y_s = Y_s if Y_s <=Ny+1 else Ny+1
    X_s = Y_s if Y_s <=Nx+1 else Nx+1
    
    index_star = [int(Y_s-1), int(X_s-1)]

    ymin = index_star[0]-1 if index_star[0]-1 >= 0 else 0 # make sure we do not collide with the edge of the image
    ymax = index_star[0]+2 if index_star[0]+2 <= Ny else Ny
    xmin = index_star[1]-1 if index_star[1]-1 >= 0 else 0 
    xmax = index_star[1]+2 if index_star[1]+2 <= Nx else Nx
    
    star_neighbourhood_flux = raw_flux[ymin:ymax, xmin:xmax] # region where star center must lie according to data header
    star_neighbourhood_indices = img_indices[ymin:ymax,xmin:xmax,:].ravel().reshape((len(img_indices[ymin:ymax, xmin:xmax, :].ravel())/2,2))
    index_star = star_neighbourhood_indices[np.argmax(star_neighbourhood_flux)]
    
    # define initial aperture and neighbourhood
    ymin = index_star[0]-1 if index_star[0]-1 >= 0 else 0 # make sure we do not collide with the edge of the image
    ymax = index_star[0]+2 if index_star[0]+2 <= Ny else Ny
    xmin = index_star[1]-1 if index_star[1]-1 >= 0 else 0 
    xmax = index_star[1]+2 if index_star[1]+2 <= Nx else Nx
    star_neighbourhood_indices = img_indices[ymin:ymax,xmin:xmax,:].ravel().reshape((len(img_indices[ymin:ymax, xmin:xmax, :].ravel())/2,2))
    
    # define the inital indices used for the aperture and the indices in the neighbourhood of the aperture
    indices_aperture = np.array([index_star])
    indices_neighbourhood = absdiff2d(indices_aperture, star_neighbourhood_indices)
    
    # get initial aperture
    aperture = np.zeros((raw_flux.shape))
    for i in range(len(indices_aperture)):
        aperture[indices_aperture[i][0],indices_aperture[i][1]] = 1

    # initialize some parameters for while loop
    N_pixel = 1 # number of pixels to used to create the aperture
    pixel_lim = pixel_lim if pixel_lim < Nx*Ny else Nx*Ny-1 #largest pixel aperture to try is leaving 1 pixel for the background
    N_pixels = []    
    std = []
    apertures = [aperture]
    index_best_aperture = -1
    while N_pixel <= pixel_lim:
        # copy raw data
        time = np.copy(raw_time)
        flux = np.copy(raw_flux)
        pixflux = np.copy(raw_pixflux)
        
        # do not do this the first time
        if N_pixel > 1:
            
            # increase the aperture, but only with pixels that connect to existing aperture
            flux_neighbourhood = np.array([flux[indices_neighbourhood[i][0],indices_neighbourhood[i][1]] for i in range(len(indices_neighbourhood))])
            img_indices_neighbourhood = np.array([img_indices[indices_neighbourhood[i][0],indices_neighbourhood[i][1]] for i in range(len(indices_neighbourhood))])
            new_pixel_index = img_indices_neighbourhood[np.argmax(flux_neighbourhood)]
            indices_aperture = np.append(indices_aperture, [new_pixel_index], axis = 0)

            # find the new neighbourhood
            ymin = new_pixel_index[0]-1 if new_pixel_index[0]-1 >= 0 else 0 # make sure we do not collide with the edge of the image
            ymax = new_pixel_index[0]+2 if new_pixel_index[0]+2 <= Ny else Ny
            xmin = new_pixel_index[1]-1 if new_pixel_index[1]-1 >= 0 else 0 
            xmax = new_pixel_index[1]+2 if new_pixel_index[1]+2 <= Nx else Nx
            
            # get new neighbourhood
            indices_new_neighbourhood = img_indices[ymin:ymax, xmin:xmax, :].ravel().reshape((len(img_indices[ymin:ymax, xmin:xmax, :].ravel())/2,2))
            indices_new_neighbourhood = diff2d(indices_new_neighbourhood, indices_aperture)
            indices_neighbourhood = diff2d(indices_neighbourhood, indices_aperture)
            indices_neighbourhood = join2d(indices_new_neighbourhood, indices_neighbourhood)
        
            # get aperture
            aperture = np.zeros((flux.shape))
            for i in range(len(indices_aperture)):
                aperture[indices_aperture[i][0],indices_aperture[i][1]] = 1
            apertures = np.append(apertures, [aperture], axis = 0)
            
        if intermediate_plot is True:
            # create an aperture for the neighbourhood to show in the plot
            neighbourhood_aperture = np.zeros((flux.shape))
            for i in range(len(indices_neighbourhood)):
                neighbourhood_aperture[indices_neighbourhood[i][0],indices_neighbourhood[i][1]] = 1
            
            # create outlines for aperture and neighbourhood
            outline = make_aperture_outline(aperture) # a new outline (ONLY for figure)
            outline2 = make_aperture_outline(neighbourhood_aperture) # a new outline (ONLY for figure)
            
            # create a plot of combined flux and aperture
            plt.imshow(flux, cmap = 'Reds_r', interpolation = 'none')
            plt.colorbar()
            plt.plot(outline[:, 0], outline[:, 1],color='b', zorder=10, lw=2.5)#,label=str(kepmag))
            plt.plot(outline2[:, 0], outline2[:, 1],color='g', zorder=10, lw=2.5, ls = '--')            
            plt.xlabel('X', size = 12)
            plt.ylabel('Y', size = 12)
            plt.xlim(0,aperture.shape[1])
            plt.ylim(0,aperture.shape[0])
            #plt.savefig(outputfolder + '/aperture_'+ str(starname) + '_perc' + str(perc[p]), dpi =300)
            plt.show()
            plt.close()
        
        # rest of Pipeline module 1: photometry
        bgflux, pixflux, time = get_backgroundflux(pixflux, aperture, time, outputfolder, starname, plot = False)
        time, flux = get_flux(pixflux, aperture, starname, bgflux, outputfolder, time, plot = False)
        time, flux, pixflux = remove_NaN_zero_inf_neg_values(time, flux, pixflux)
        
        # Pipeline module 2: correct for motion of the satelite
        Xcof, Ycof = get_centerofflux(pixflux, time, aperture, outputfolder, starname, plot = False)
        for i in range(3):
            time, flux, Xcof, Ycof = remove_thrusteronpoints(time, flux, Xcof, Ycof, outputfolder, starname, plot = False)
        #time, flux = remove_NaN_zero_inf_neg_values(time, flux)
        time, flux = remove_poyntingjitter(time, flux, Xcof, Ycof, outputfolder, starname, plot = False)
        time, flux, Xcof, Ycof = remove_outliers(time, flux, Xcof, Ycof, outputfolder, starname, plot = False)
        
        # get average background noise
        outliers_rm_time = np.delete(time, np.where(np.abs(flux-np.mean(flux)) > 3*np.std(flux))) # renmove 3-sigma outliers
        outliers_rm_flux = np.delete(flux, np.where(np.abs(flux-np.mean(flux)) > 3*np.std(flux))) # renmove 3-sigma outliers
        N_pixels = np.append(N_pixels, N_pixel)    
        std = np.append(std, np.std(outliers_rm_flux))
        
        # next round we add another pixel
        N_pixel +=1

    # best_aperture
    index_best_aperture = np.argmin(std)
    best_aperture = apertures[index_best_aperture,:,:]    
    
    if plot is True:
            # create a plot relating standard deviation to the size of the aperture
            plt.plot(N_pixels,std,color='k',lw=2)
            plt.axvline(N_pixels[index_best_aperture], lw = 1, ls = '--', color ='b')
            plt.annotate(r'N$_{best}=$' + str(N_pixels[index_best_aperture]), xy = (N_pixels[index_best_aperture]/len(N_pixels),0.95), ha = 'center', xycoords="axes fraction", size =12, color = 'b')
            plt.xlabel('N pixels used', size =12)
            plt.ylabel('Standard deviation', size = 12)
            plt.savefig(outputfolder + '/stdNpixel_'+ str(starname), dpi =300)
            #plt.show()
            plt.close()
        
            # create outlines for aperture and neighbourhood
            outline = make_aperture_outline(best_aperture) # a new outline (ONLY for figure)            
            
            # create a plot of combined flux and aperture
            plt.imshow(raw_flux, cmap = 'Reds_r', interpolation = 'none')
            plt.colorbar()
            plt.plot(outline[:, 0], outline[:, 1],color='b', zorder=10, lw=2.5)#,label=str(kepmag))
            plt.xlabel('X', size = 12)
            plt.ylabel('Y', size = 12)
            plt.xlim(0,best_aperture.shape[1])
            plt.ylim(0,best_aperture.shape[0])
            plt.savefig(outputfolder + '/aperture_'+ str(starname), dpi =300)
            #plt.show()
            plt.close()
            
    return best_aperture

def get_backgroundflux (pixflux, aperture, time, outputfolder, starname, campaign, cutoff = 2, plot = True):
    """
    Find the mean background flux per pixel for all time. This done by looking
    at all flux outside of the aperture, removing outliers (potential neighbouring stars)
    and taking the median.
    """

    # only for the first campaigns background flux has not been subtracted in the pipeline already
    if campaign in [1,2]:
        # add pixflux over complete data set
        flux = np.nansum(pixflux, axis = 0)
    
        # this part estimates the background outside of the aperture
        bg_aperture = (aperture == 0)*1
        bg_aperture_outlier_rm = (flux < cutoff * np.median(flux))*1
        bg_aperture = bg_aperture_outlier_rm * bg_aperture
        if np.nansum(bg_aperture) == 0: bg_aperture = (aperture == 0)*1
        
        # estimated background flux from the values in the aperture
        bgflux = np.array([np.nanmedian( (pixflux[i] * bg_aperture)[np.nonzero(pixflux[i] * bg_aperture)]) for i in range(0,len(pixflux))])
        
        # create a simple flux of the background flux as function of time
        if plot is True:
            plt.title('Background flux', size = 15)
            plt.scatter(time, bgflux, color = 'b', s=0.5)
            plt.xlabel('Time [BKJD]', size = 12)
            plt.ylabel(r'Background flux [pixel$^{-1}$]', size = 12)
            plt.savefig(outputfolder + '/bgflux_'+ str(starname), dpi =300)
            plt.close()
    
    else:
        bgflux = np.array([0.0 for i in range(0,len(pixflux))])
    
    return bgflux, pixflux, time

def get_flux (pixflux, aperture, starname, bgflux, outputfolder, time = '', plot = True):
    """
    Find total star flux for all time, subtract background and return
    flux, which has not yet been corrected for any systematic errors.
    """

    # determine the starflux
    starflux = [np.nansum(pixflux[i] * aperture) for i in range(0, len(pixflux))] # add all pixflux within aperture
    area = np.sum(aperture) # aperture area
    flux = np.array([starflux[i] - bgflux[i]*area for i in range(0, len(pixflux))]) # remove background flux

    # create a simple flux of the star flux as function of time
    if plot is True:
        plt.title('Raw flux', size = 15)
        plt.scatter(time, flux, color = 'k', s=0.5)
        # time should be provided if one wants to plot
        plt.xlabel('Time [BKJD]', size = 12)
        #plt.xlim(1980,1985)
        #plt.ylim(176000,176500)
        plt.ylabel('Star flux [arbitrary units]', size = 12)
        plt.savefig(outputfolder + '/lc_'+ str(starname), dpi =300)
        plt.close()
    
    return time, flux

def make_aperture_outline(frame, no_combined_images=1, threshold=0.5):
    ## this is a little module that defines so called outlines to be used for plotting apertures

    thres_val = no_combined_images * threshold
    mapimg = (frame > thres_val)
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))


    segments = np.array(l)

    x0 = -0.5
    x1 = frame.shape[1]+x0
    y0 = -0.5
    y1 = frame.shape[0]+y0

    #   now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
    segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

    return segments

def remove_NaN_zero_inf_neg_values(time, flux, pixflux):
    """
    Removes NaN values from your array.
    """
    
    # remove NaN values
    time = np.delete(time, np.where(np.isnan(flux)))
    pixflux = np.delete(pixflux, np.where(np.isnan(flux)), axis = 0)
    flux = np.delete(flux, np.where(np.isnan(flux)))
    
    # remove zero and negative values
    time = np.delete(time, np.where(flux <= 0))
    pixflux = np.delete(pixflux, np.where(flux <= 0), axis = 0)
    flux = np.delete(flux, np.where(flux <= 0))
    
    # remove infinte values
    time = np.delete(time, np.isinf(flux))
    pixflux = np.delete(pixflux, np.isinf(flux), axis = 0)
    flux = np.delete(flux, np.isinf(flux))
    
    # remove NaN values also for time
    pixflux = np.delete(pixflux, np.where(np.isnan(time)), axis = 0)
    flux = np.delete(flux, np.where(np.isnan(time)))
    time = np.delete(time, np.where(np.isnan(time)))
    
    return time, flux, pixflux
 
def get_aperture_of_fixed_size(time, pixflux, X_s, Y_s, inputfolder, outputfolder, filename, starname, aper_size, pixel_lim = 25, plot = True, intermediate_plot = False):
    """
    Returns aperture for fixed size of aper_size pixels.
    """
    
    # get raw data
    raw_time = np.copy(time)
    raw_pixflux = np.copy(pixflux)

    # get added fluc values and array containing the indices
    raw_flux = np.nansum(raw_pixflux, axis = 0)

    # get dimensions and indices array of rawflux
    Nx, Ny = raw_flux.shape
    Nx += 1
    Ny += 1
    img_indices = np.array([[i,j] for i in range(raw_flux.shape[0]) for j in range(raw_flux.shape[1])]).reshape((raw_flux.shape[0],raw_flux.shape[1],2))
    
    # use FITS file position as star center estimate
    Y_s = Y_s if Y_s >=1 else 1 # make sure pixel values are inside the image
    X_s = X_s if X_s >=1 else 1
    Y_s = Y_s if Y_s <=Ny+1 else Ny+1
    X_s = Y_s if Y_s <=Nx+1 else Nx+1
    
    index_star = [int(Y_s-1), int(X_s-1)]

    ymin = index_star[0]-1 if index_star[0]-1 >= 0 else 0 # make sure we do not collide with the edge of the image
    ymax = index_star[0]+2 if index_star[0]+2 <= Ny else Ny
    xmin = index_star[1]-1 if index_star[1]-1 >= 0 else 0 
    xmax = index_star[1]+2 if index_star[1]+2 <= Nx else Nx
    
    star_neighbourhood_flux = raw_flux[ymin:ymax, xmin:xmax] # region where star center must lie according to data header
    star_neighbourhood_indices = img_indices[ymin:ymax,xmin:xmax,:].ravel().reshape((len(img_indices[ymin:ymax, xmin:xmax, :].ravel())/2,2))
    index_star = star_neighbourhood_indices[np.argmax(star_neighbourhood_flux)]
    
    # define initial aperture and neighbourhood
    ymin = index_star[0]-1 if index_star[0]-1 >= 0 else 0 # make sure we do not collide with the edge of the image
    ymax = index_star[0]+2 if index_star[0]+2 <= Ny else Ny
    xmin = index_star[1]-1 if index_star[1]-1 >= 0 else 0 
    xmax = index_star[1]+2 if index_star[1]+2 <= Nx else Nx
    star_neighbourhood_indices = img_indices[ymin:ymax,xmin:xmax,:].ravel().reshape((len(img_indices[ymin:ymax, xmin:xmax, :].ravel())/2,2))
    
    # define the inital indices used for the aperture and the indices in the neighbourhood of the aperture
    indices_aperture = np.array([index_star])
    indices_neighbourhood = absdiff2d(indices_aperture, star_neighbourhood_indices)
    
    # get initial aperture
    aperture = np.zeros((raw_flux.shape))
    for i in range(len(indices_aperture)):
        aperture[indices_aperture[i][0],indices_aperture[i][1]] = 1

    # initialize some parameters for while loop
    N_pixel = 1 # number of pixels to used to create the aperture
    pixel_lim = pixel_lim if pixel_lim < Nx*Ny else Nx*Ny-1 #largest pixel aperture to try is leaving 1 pixel for the background
    N_pixels = []    
    std = []
    apertures = [aperture]
    index_best_aperture = -1
    while N_pixel <= aper_size:
        # copy raw data
        time = np.copy(raw_time)
        flux = np.copy(raw_flux)
        pixflux = np.copy(raw_pixflux)
        
        # do not do this the first time
        if N_pixel > 1:
            
            # increase the aperture, but only with pixels that connect to existing aperture
            flux_neighbourhood = np.array([flux[indices_neighbourhood[i][0],indices_neighbourhood[i][1]] for i in range(len(indices_neighbourhood))])
            img_indices_neighbourhood = np.array([img_indices[indices_neighbourhood[i][0],indices_neighbourhood[i][1]] for i in range(len(indices_neighbourhood))])
            new_pixel_index = img_indices_neighbourhood[np.argmax(flux_neighbourhood)]
            indices_aperture = np.append(indices_aperture, [new_pixel_index], axis = 0)

            # find the new neighbourhood
            ymin = new_pixel_index[0]-1 if new_pixel_index[0]-1 >= 0 else 0 # make sure we do not collide with the edge of the image
            ymax = new_pixel_index[0]+2 if new_pixel_index[0]+2 <= Ny else Ny
            xmin = new_pixel_index[1]-1 if new_pixel_index[1]-1 >= 0 else 0 
            xmax = new_pixel_index[1]+2 if new_pixel_index[1]+2 <= Nx else Nx
            
            # get new neighbourhood
            indices_new_neighbourhood = img_indices[ymin:ymax, xmin:xmax, :].ravel().reshape((len(img_indices[ymin:ymax, xmin:xmax, :].ravel())/2,2))
            indices_new_neighbourhood = diff2d(indices_new_neighbourhood, indices_aperture)
            indices_neighbourhood = diff2d(indices_neighbourhood, indices_aperture)
            indices_neighbourhood = join2d(indices_new_neighbourhood, indices_neighbourhood)
            
        # next round we add another pixel
        N_pixel +=1
        
    # get aperture
    best_aperture = np.zeros((flux.shape))
    for i in range(len(indices_aperture)):
        best_aperture[indices_aperture[i][0],indices_aperture[i][1]] = 1
            
    if plot is True:
        # create outlines for aperture and neighbourhood
        outline = make_aperture_outline(best_aperture) # a new outline (ONLY for figure)   
        
        # create a plot of combined flux and aperture
        plt.imshow(raw_flux, cmap = 'Reds_r', interpolation = 'none')
        plt.colorbar()
        plt.plot(outline[:, 0], outline[:, 1],color='b', zorder=10, lw=2.5)#,label=str(kepmag))
        plt.xlabel('X', size = 12)
        plt.ylabel('Y', size = 12)
        plt.xlim(0,best_aperture.shape[1])
        plt.ylim(0,best_aperture.shape[0])
        plt.savefig(outputfolder + '/aperture_'+ str(starname), dpi =300)
        #plt.show()
        plt.close()
        
    return best_aperture
    
def remove_time_range(time, flux, pixflux, t_min, t_max):
    """
    Remove all data points in a certain time range (t_min, t_max), which is 
    known to contain bad flux values.
    """
    
    # get indices in time range (t_min, t_max)
    indices = [i for i in range(0, len(time)) if t_min <= time[i] <= t_max]
    
    # remove these points
    time = np.delete(time, indices)
    pixflux = np.delete(pixflux, indices, axis = 0)
    flux = np.delete(flux, indices)    
    
    return time, flux, pixflux
    
    
def from_pixel_to_flux (starname, filename, inputfolder, outputfolder, aperture):
    """
    Goes from the original pixelfiles to a flux, corrected for systematic errors
    by the K-2 instruments and satelite.
    """

    time, pixflux, magnitude, X_s, Y_s, campaign = get_pixelpixflux(starname, filename, inputfolder, outputfolder, remove = True)

    bgflux, pixflux, time = get_backgroundflux(pixflux, aperture, time, outputfolder, starname, campaign, plot = True)

    time, flux = get_flux(pixflux, aperture, starname, bgflux, outputfolder, time, plot = True)

    time, flux, pixflux = remove_NaN_zero_inf_neg_values(time, flux, pixflux)
    
    # save raw time and flux
    data = np.array(zip(time, flux))
    data_header = '0 time 1 flux'
    np.savetxt(os.path.join(outputfolder,'rawlc_' 
               + str(starname) + '.txt'), data, header=data_header)

    return time, pixflux, flux, aperture