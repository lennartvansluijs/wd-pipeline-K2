"""
K2 pipeline: searching for exoplanets around white dwarfs.

Input: K2 pixeltarget file data for a bunch of stars in the inputfolder.
Output: List of Objects of Interst (OIs), with possible transits.

This script will look at 7 already discovered planets by the K2 satelite.

Code by Lennart van Sluijs
"""

import os
import numpy as np
from astropy.io import fits as pyfits
from pixeltoflux import from_pixel_to_flux
from correctformotion import correct_for_satelite_motion
from findperiod import find_period
import matplotlib.pyplot as plt

print '-----------------------------------------------------------------------'
print '-----------------------------------------------------------------------'
print '      K2 pipeline -  searching for exoplanets around White Dwarfs      '
print '                                                                       '
print '                            (long cadence)                            '
print ''
print '                          by Lennart van Sluijs                        '
print '-----------------------------------------------------------------------'
print '-----------------------------------------------------------------------'
print ' '

# reduce for short cadence instead of long cadence data
inputfolder = '../lc_data/'
outputpath = '../lc_output/'

# create list of all files that need to be processed
fnames = [f for f in os.listdir(inputfolder) if os.path.isfile(os.path.join(inputfolder, f))]

print 'Files that will be processed:'
for p in range(0,len(fnames)): print fnames[p]
print ''
print 'Now processing: '

results = np.empty((len(fnames),15))
for f in range(0,len(fnames)):
    
    # get filnemae and starname of current object
    if 'c11' in fnames[f]:
        filename = fnames[f]
        starname = filename[4:18] # only use this one in the special case of campaign 11 where data was split into two parts
    else:
        filename = fnames[f]
        starname = filename[4:13]
        
    print filename + ' (' + str(f+1) + '/' + str(len(fnames)) + ')'
    
    #create outputfolder for this star
    outputfolder = os.path.join(outputpath,str(starname))
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # Pipeline module 1: perform circular aperture photometry
    time, pixflux, flux, aperture = from_pixel_to_flux(starname, filename, inputfolder, outputfolder)
    
    # save all the before correcting for the poynting jitter
    data = np.array(zip(time,flux))
    data_header = '1 time 2 flux'
    #np.savetxt(outputfolder + '/injected_lc_'+ str(starname), data, header = data_header)
    np.save(outputfolder + '/pixflux_'+ str(starname), pixflux)
    np.save(outputfolder + '/aperture_'+ str(starname), aperture)

    # Pipeline module 2: correct for motion of the satelite
    time, flux = correct_for_satelite_motion(time, pixflux, flux, aperture, outputfolder, starname)
    
    # use this when performing BLs on the combined c11 lightcurves
    data = np.loadtxt(os.path.join(outputfolder,'systematiccorrectedlc_' + str(starname) + '.txt'))
    time = np.array(data[:,0])
    flux = np.array(data[:,1])
    
    # Pipeline module 3: find the best fitting period
    P, BLS, params, SDE = find_period(time, flux, outputfolder, starname)
    

    # we want to save all relevant information
    params = params.ravel()

    for i in range(len(params)):
        results[f, i] = params[i]
  
print ''
print 'Saving results.'
print ''

# save results
hdu = pyfits.PrimaryHDU(results)
hdu.writeto(os.path.join(outputpath, 'results.fits'), clobber=True)