"""
Some functions which can create transit models.

Code by Lennart van Sluijs
"""

from model_transits import occultuniform, occultquad, t2z
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from matplotlib.pyplot import cm 

def fold_data (time, flux, period):
    """
    Fold data for a given period.
    """
    folded = (time % period)
    inds = np.array(folded).argsort()
    time_folded = folded[inds]
    flux_folded = flux[inds]
    
    return time_folded, flux_folded

def inject_periodic_box_signal(time, flux, P, q, delta, plot = True):
    """
    Injects an artifical transit signal by assuming the transit signal can
    approximately be described by a periodic box signal.
    The epoch of the signal will be chosen randomly.
    
    Input:
    time = array containing the time series for which box signal must be evaluated
    flux = flux of real K2 data corresponding to time series
    P = period of injected signal.
    q = relative transit depth of injected signal.
    delta = relative transit depth of the signal.
    
    Output:
    flux_injected = flux with injected transit signal.
    """
    
    # randomly pick an epoch somewhere in the time series
    time_min = time[0]
    time_max = time[-1]
    epoch = random.uniform(time_min, time_max)
    t_s0 = epoch # start of transit time,
    #here we assume epoch corresponds with the start of a transit
    t_e0 = epoch + q * P # end of transit time
    
    # lenght of time array
    N = len(time)
    
    # calculate all start of transits in time series
    n_min = int( (time_min - t_s0)/P ) - 1
    n_max = int( (time_max - t_s0)/P ) + 1
    t_s = [ t_s0 + P * n for n in range(n_min,n_max) ]
    
    # calculate all end of transits in time series
    n_min = int( (time_min - t_e0)/P ) + 1
    n_max = int( (time_max - t_e0)/P ) - 1
    t_e = [ t_e0 + P * n for n in range(n_min,n_max) ]
    
    # evaluate periodic box signal in time series
    signal = np.zeros(len(time)) + 1 # signal to be injected
    for i in range(0, len(time)):
        for j in range(0, len(t_s)):
            if t_s[j] <= time[i] <= t_s[j] + P * q:
                signal[i] = 1 - delta
        for j in range(0, len(t_e)):
            if t_e[j] - P * q <= time[i] <= t_e[j]:
                signal[i] = 1 - delta
    
    # inject signal
    flux_injected = flux * signal
    
    if plot is True:
        fig = plt.figure(figsize=(14,5))
        gs = gridspec.GridSpec(1, 3)
        #gs.update(wspace = 1, hspace = 1)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[0,2])
        ax1.scatter(time, signal, color = 'b')
        ax1.set_xlabel('Time [BKJD]', size = 12)
        ax1.set_ylabel('Flux [arbitrary units]', size = 12)
        ax2.scatter(time, flux, color = 'k')
        ax2.set_xlabel('Time [BKJD]', size = 12)
        ax3.scatter(time, flux_injected, color = 'g')
        ax3.set_xlabel('Time [BKJD]', size = 12)
        plt.show()
    
    return flux_injected
    
"""
def inject_Mandol_Agol(time, flux, pixflux, N_t, R_p, P, R_s, M_s, outputfolder, starname, e = 0, plot = True):

    Creates a Mandol and Agol model of a lightcurve and injects this model into
    a K2 lightcurve.
    
    Part of this code is created by Ian Crossfield based on the Mandol & Agol
    (2002) transit signal models.
    
    Input parameters:
    time = observed time series
    flux = observed flux series
    N_t number of data points in artifical time series
    R_p radius of the planet
    R_s radius of the star
    P = period of the planet (in same units as time series, here in days)
    i = orbital inclination
    e = eccentricity
    
    Output:
    time_model = array containing time series of the model
    flux_model = array containing flux series of the model

    
    # calculate a semi-major axis from the period
    P_sec = P * 86400 # calculate the period in seconds
    G = 6.67408e-11 # graviational constant in SI units
    a = ( (G * P_sec**2 * M_s)/(4 * np.pi**2))**(1./3.)
    
    # create artificial time series
    t_s = np.min(time) - 1 # create some extra margin for the first data point
    t_e = np.max(time)
    time_model = np.linspace(t_s, t_e, N_t)
    time_model_min = time_model[0]
    time_model_max = time_model[-1]
    
    # pick a random epoch
    epoch = random.uniform(time_model_min, time_model_max) # ephermeris is random
    
    # pick a random inclination
    b_max = (R_s + R_p)/R_s # maximal impact parameter if we just want to have a grazing transit
    b = random.uniform(0,b_max) # pick a impact parameter between 0 and b_max
    i = np.arccos( (b * R_s)/a ) * 57.2957795 # calculate the inclination from b
    
    # go to new coordinate z, the transit crossing parameter
    z = t2z(tt = epoch, per = P, inc = i, hjd = time_model, ars = a/R_s, ecc = e) 
    
    # go to a transit curve
    gamma = [-0.0097,0.3679] # lim darkening coefficients according to Gianninas et al. (2013) for Teff = 5000 K Agol (2011) and log g = 6.5
    flux_model = occultquad(z, R_p/R_s, gamma)
    flux_model = np.array([0 if flux_model[i] < 0 else flux_model[i] for i in range(len(flux_model))]) # modelling has some issues if R_p > R_s, this fixes the negative flux values
    
    # dillute model
    exptime = 30./1440. # 30/1440 min is exptime in days
    dilluted_flux_model = np.copy(flux)    
    for i in range(len(time)):
        indices = np.where(np.logical_and(time_model>=time[i]-exptime, time_model<=time[i])) # incices of 
        dilluted_flux_model[i] = np.mean(flux_model[indices])
    
    # inject dilluted flux model
    flux_injected = dilluted_flux_model * flux

    # plot if plot is True
    if plot is True:
        plt.close()
        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(2, 2)
        #gs.update(wspace = 1, hspace = 1)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[1,0])
        ax4 = plt.subplot(gs[1,1])
        ax1.set_title('Observed flux', size = 15)
        ax1.scatter(time, flux, color = 'k')
        ax1.set_xlabel('Time [BKJD]', size = 12)
        ax1.set_ylabel('Flux [arbitrary units]', size = 12)
        ax2.set_title('Transit model', size = 15)
        ax2.plot(time_model, flux_model, color = 'k')
        ax2.set_xlabel('Time [BKJD]', size = 12)
        #ax2.set_ylim(np.min(flux_model), np.max(flux_model))
        ax3.set_title('Dilluted transit model', size = 15)
        ax3.scatter(time, dilluted_flux_model, color = 'k')
        #ax3.set_ylim(np.min(dilluted_flux_model), np.max(dilluted_flux_model))
        ax3.set_xlabel('Time [BKJD]', size = 12)
        ax4.set_title('Model injected observed flux', size = 15)
        ax4.scatter(time, flux_injected, color = 'k')
        ax4.set_xlabel('Time [BKJD]', size = 12)
        #plt.tight_layout()
        plt.savefig(outputfolder + '/injected_'+ str(starname))
        #plt.show()
        plt.close()
    
    return time, flux_injected, pixflux
"""

def inject_transit(time, flux, R_p, P, R_s, M_s, outputfolder, starname, exptime = 1./1440., e = 0, plot = True, N_t = 5000):
    """
    Get quadratic occult transitmodel given the input parameters.
    
    INPUT PARAMETERS:
    time - input array containing time information
    params - parameters [epoch, b, R_p, P, gamma_1, gamma_2, R_s, M_s, e]
    
    DEFAULT PARAMETERS:
    diluted - if True dilute by given exptime
    paramsclass - if True use paramsclass parameters instead of normal parameters
    exptime - exptime used for diluting the lightcurve
    N_t - resolution of oversampling in case of diluting the lightcurve
    
    OUTPUT PARAMETERS:
    model - best Mandol and Agol model
    diluted_model - best diluted Mandol and Agol model
    """
    
    # calculate a semi-major axis from the period
    P = P*2 # somehow this seems necessary
    P_sec = P * 86400 # calculate the period in seconds
    G = 6.67408e-11 # graviational constant in SI units
    a = ( (G * P_sec**2 * M_s)/(4 * np.pi**2))**(1./3.)
    gamma = [-0.0097,0.3679]
    
    # pick a random epoch
    epoch = random.uniform(time[0], time[-1]) # ephermeris is random
    
    # pick a random inclination
    b_max = (R_s + R_p)/R_s # maximal impact parameter if we just want to have a grazing transit
    b = random.uniform(0,b_max) # pick a impact parameter between 0 and b_max
    b = 0
    i = np.arccos( (b * R_s)/a ) * 57.2957795 # calculate the inclination from b
    
    # create oversampled lightcurve first for one period
    t_s = np.min(0) # create some extra margin for the first data point
    t_e = np.max(P/2)
    time_model = np.linspace(t_s, t_e, N_t)
    
    # go to new coordinate z, the transit crossing parameter
    z = t2z(tt = epoch, per = P, inc = i, hjd = time_model, ars = a/R_s, ecc = e) 
    
    # go to a transit curve
    model = occultquad(z, R_p/R_s, gamma)
    model = np.array([1e-10 if model[i] < 0 else model[i] for i in range(len(model))]) # modelling has some issues if R_p > R_s, this fixes the negative flux values
    model = model
    
    plt.scatter(time_model,model)
    plt.show()
    
    # bak to original period
    P = P/2.
        
    # create a oversampled model lightcurve of twice the period
    diluted_model = np.copy(time)
    time_model_f, model_f = fold_data(time_model, model, P)
    time_model_t = np.append(time_model_f,time_model_f+P)
    model_t = np.append(model_f,model_f)
    
    # find timestamps in range (t-exptime/2,t+exptime/2) in folded time series
    t_s = (time-exptime/2.)%P
    t_e = t_s + exptime
    i_s = [np.argmin(np.abs(time_model_t-t_s[i])) for i in range(len(t_s))] # corresponding indices
    i_e = [np.argmin(np.abs(time_model_t-t_e[i])) for i in range(len(t_e))]
    
    # dilute the oversampled model
    diluted_model = np.array([ np.mean(model_t[np.int(i_s[j]):np.int(i_e[j])]) for j in range(len(i_s))])
    
    # inject dilluted flux model
    flux_injected = diluted_model * flux

    # plot if plot is True
    if plot is True:
        plt.close()
        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(2, 2)
        #gs.update(wspace = 1, hspace = 1)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[1,0])
        ax4 = plt.subplot(gs[1,1])
        ax1.set_title('Observed flux', size = 15)
        ax1.scatter(time, flux, color = 'k')
        ax1.set_xlabel('Time [BKJD]', size = 12)
        ax1.set_ylabel('Flux [arbitrary units]', size = 12)
        ax2.set_title('Transit model', size = 15)
        ax2.scatter(time_model, model, color = 'k')
        ax2.set_xlabel('Folded time [BKJD]', size = 12)
        ax2.set_ylim(np.min(model), np.max(model))
        ax3.set_title('Diluted transit model', size = 15)
        ax3.scatter(time, diluted_model, color = 'k')
        #ax3.set_ylim(np.min(dilluted_flux_model), np.max(dilluted_flux_model))
        ax3.set_xlabel('Time [BKJD]', size = 12)
        ax4.set_title('Model injected observed flux', size = 15)
        ax4.scatter(time, flux_injected, color = 'k')
        ax4.set_xlabel('Time [BKJD]', size = 12)
        #plt.tight_layout()
        #plt.savefig(outputfolder + '/injected_'+ str(starname))
        plt.show()
        plt.close()
    
    return time, flux_injected
    
def inject_Mandol_Agol(time, flux, R_p, P, R_s, M_s, outputfolder, starname, exptime = 1./1440., e = 0, plot = True, N_t = 5000):
    """
    Get quadratic occult transitmodel given the input parameters.
    
    INPUT PARAMETERS:
    time - input array containing time information
    params - parameters [epoch, b, R_p, P, gamma_1, gamma_2, R_s, M_s, e]
    
    DEFAULT PARAMETERS:
    diluted - if True dilute by given exptime
    paramsclass - if True use paramsclass parameters instead of normal parameters
    exptime - exptime used for diluting the lightcurve
    N_t - resolution of oversampling in case of diluting the lightcurve
    
    OUTPUT PARAMETERS:
    model - best Mandol and Agol model
    diluted_model - best diluted Mandol and Agol model
    """
    
    # calculate a semi-major axis from the period
    P = P*2 # somehow this seems necessary
    P_sec = P * 86400 # calculate the period in seconds
    G = 6.67408e-11 # graviational constant in SI units
    a = ( (G * P_sec**2 * M_s)/(4 * np.pi**2))**(1./3.)
    gamma = [-0.0097,0.3679]
    T_max = 30./1440. # maximum transit duration
    
    # pick a random epoch
    epoch = random.uniform(time[0], time[0]+P/2.) # epoch is random
    
    pick = True
    while pick: # check if not in 'danger zone' regarding the epoch pick
        epoch = random.uniform(time[0], time[0]+P/2.) # epoch is random
        x1 = (epoch - exptime - T_max/2.)%(P/2.)
        x2 = (epoch + exptime + T_max/2.)%(P/2.)
        if x1 < x2:
            pick = False
    
    # pick a random inclination
    b_max = (R_s + R_p)/R_s # maximal impact parameter if we just want to have a grazing transit
    b = random.uniform(0,b_max) # pick a impact parameter between 0 and b_max
    i = np.arccos( (b * R_s)/a ) * 57.2957795 # calculate the inclination from b
    
    # get times when transit happen
    t_transits = np.arange(epoch, time[-1], P/2.)
    t_s = t_transits - exptime/2. - T_max/2.
    t_e = t_transits + exptime/2. + T_max/2.
    mask = np.array([True for k in range(len(time))])
    for j in range(len(t_s)):
        mask = ~np.logical_and(time>t_s[j], time<t_e[j]) * mask # these are
        # the indices for which we have to dilute the transit
    mask = ~mask
    
    # create oversampled lightcurve first for one period
    P = P/2
    t_s = epoch - exptime - T_max/2.# (epoch+P/4.-exptime/2.-T_max/2.)%P/2.#(epoch-P/2.-exptime/2.-T_max/2.)%P/2.
    t_e = epoch + exptime + T_max/2. #t_s + T_max+exptime #P/2#t_s + exptime + T_max
    margin = exptime
    time_model = np.linspace(t_s-margin, t_e+margin, N_t)
    P = P*2

    # go to new coordinate z, the transit crossing parameter
    z = t2z(tt = epoch, per = P, inc = i, hjd = time_model, ars = a/R_s, ecc = e) 
    
    # go to a transit curve
    model = occultquad(z, R_p/R_s, gamma)
    model = np.array([0 if model[i] < 0 else model[i] for i in range(len(model))]) # modelling has some issues if R_p > R_s, this fixes the negative flux values

    # back to original period
    P = P/2.
        
    # create a oversampled model lightcurve of twice the period
    diluted_model = np.array([1. for i in range(len(time))])
    time_model_f, model_f = fold_data(time_model, model, P)

    # find timestamps in range (t-exptime/2,t+exptime/2) in folded time series
    t_s = (time[mask]-exptime/2.)%P
    t_e = t_s + exptime
    i_s = [np.argmin(np.abs(time_model_f-t_s[i])) for i in range(len(t_s))] # corresponding indices
    i_e = [np.argmin(np.abs(time_model_f-t_e[i])) for i in range(len(t_e))]
    
    # dilute the oversampled model
    diluted_model[mask] = np.array([ np.mean(model_f[np.int(i_s[j]):np.int(i_e[j])]) for j in range(len(i_s))])
    
    # inject dilluted flux model
    flux_injected = diluted_model * flux
    
    # plot if plot is True
    if plot is True:
        plt.close()
        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(2, 2)
        #gs.update(wspace = 1, hspace = 1)
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[1,0])
        ax4 = plt.subplot(gs[1,1])
        ax1.set_title('Observed flux', size = 15)
        ax1.scatter(time, flux, color = 'k', s=0.5)
        ax1.set_xlabel('Time [BKJD]', size = 12)
        ax1.set_ylabel('Flux [arbitrary units]', size = 12)
        ax2.set_title('Transit model', size = 15)
        ax2.scatter(time_model, model, color = 'k', s=0.5)
        ax2.annotate('b = '+ str(round(b,3)) + '\nbmax = ' + str(round(b_max,3)) + '\nepoch ' + str(round(epoch,3)), xy = (0.1,0.1),xycoords="axes fraction")
        ax2.set_xlabel('Folded time [BKJD]', size = 12)
        ax2.set_ylim(np.min(model), np.max(model))
        ax3.set_title('Diluted transit model', size = 15)
        ax3.scatter(time, diluted_model, color = 'k', s=0.5)
        #ax3.set_ylim(np.min(dilluted_flux_model), np.max(dilluted_flux_model))
        ax3.set_xlabel('Time [BKJD]', size = 12)
        ax4.set_title('Model injected observed flux', size = 15)
        ax4.scatter(time, flux_injected, color = 'k', s=0.5)
        ax4.set_xlabel('Time [BKJD]', size = 12)
        #plt.tight_layout()
        plt.savefig(outputfolder + '/injected_'+ str(starname))
        #plt.show()
        plt.close()
        
    return time, flux_injected
    #return diluted_model, b, epoch # if we want to create a nice injection plot

""" Nice injection plot
# important standard units
R_Ceres = 473e3 # m
R_Moon = 1.737e6 # m
R_Mercurius = 2.440e6 # m
R_Earth = 6.371e6 # m
R_Mars = 3.390e6 # m
R_Jupiter = 69911e3 # m
R_Sun = 6.955e8 # m
au = 149597870700 # m
P_Earth = 365.25 # d
M_Sun = 1.9891e30 # kg

# parameters for Kepler-7b   
R_p = 1.478 * R_Jupiter * 0.5
R_s = 1.843 * R_Sun
a = 0.06224 * au
M_s = 1.347 * M_Sun
P = 4.885525 # d

# typical White Dwarf parameters
R_WD = 0.01 * R_Sun # Hansen (2004)
M_WD = 0.6 * M_Sun # Agol (2011)
G = 6.67408e-11

# range of periods and radii that should be investigated
R_min = 0.5 * R_Ceres #0.9 * R_Moon # 0.1 Ceres sized objects
R_max = 1.3 * R_Jupiter # still planets and not brown dwarfs
R_p = np.logspace(np.log10(R_min), np.log10(R_max), num = 10)
#R_p = np.array([R_Ceres])
P_min = 5./24. # d, is approximately +- 5 hour (5 hour is Roche period)
P_max = 81 # total observed time during survey
P = np.logspace(np.log10(P_min), np.log10(P_max), num = 10)

#time = np.array([1,2,3,4,5,8])
#time_model = np.linspace(1,8,100)
#model = np.array([1 for i in range(time_model)])

P = 1./24.
time = np.arange(270,353,1./1440.)
flux = np.array([1 for i in range(len(time))])
outputfolder = 'bla'
starname='bla'
time, flux = inject_Mandol_Agol(time, flux, 16*R_Earth, P, R_WD, M_WD, outputfolder, starname, exptime = 1./1440.)
time_f, flux_f = fold_data(time, flux, P)
print np.float(len(flux_f[np.where(flux_f<1)]))/np.float(len(flux_f))
plt.plot(time_f, flux_f)
plt.show()

#print (R_Earth/R_WD)**2
P = 11./24.
n=10
colors= cm.nipy_spectral(np.linspace(0,1,n))
#diluted_model, b, epoch = inject_Mandol_Agol(time, flux, R_Earth, P, R_WD, M_WD, outputfolder, starname, exptime = 30./1440.)

inject_Mandol_Agol(time, flux, R_Earth, P, R_WD, M_WD, outputfolder, starname, exptime = 30./1440.)

fig, ax = plt.subplots()
for i in range(n):
    diluted_model, b, epoch = inject_Mandol_Agol(time, flux, R_Earth, P, R_WD, M_WD, outputfolder, starname, exptime = 30./1440.)
    time_f, diluted_model_f = fold_data(time,diluted_model,P)
    plt.plot(time_f, diluted_model_f, lw=2,color=colors[i])
    plt.annotate(str(np.round(b,2)),xy = (epoch%(P),np.min(diluted_model)-0.0015), color =colors[i], ha = 'center')
plt.axhline(1,lw=1,color='k')
plt.ylim(0.965,1.0025)
plt.yticks(np.linspace(0.965,1.0025,5))
ax.set_yticklabels(np.round(np.linspace(0.965,1.0025,5),3))
plt.xticks(np.linspace(0,P,5))
ax.tick_params(axis='both', direction='in')
ax.set_xticklabels(np.round(np.linspace(0,P,5),2))
plt.ylabel('Relative flux', size = 12)
plt.xlabel('Folded time [BKJD]', size = 12)
plt.xlim(np.min(time_f),np.max(time_f))
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
plt.savefig('random_injection_2.png',dpi=300)
plt.show()

"""