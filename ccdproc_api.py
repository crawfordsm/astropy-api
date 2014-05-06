#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from astropy.nddata import NDData
from astropy.io import fits
from astropy import units as u
from astropy.stats.funcs import sigma_clip

from ccdproc import ccddata
from ccdproc import ccdproc

'''
The ccdproc package provides tools for the reduction and
analysis of optical data captured with a CCD.   The package
is built around the CCDData class, which has built into
it all of the functions to process data.  The CCDData object
contains all the information to describe the 2-D readout
from a single amplifier/detector.

The CCDData class inherits from the NDData class as its base object
and the object on which all actions will be performed.  By
inheriting from the CCD data class, basic array manipulation
and error handling are already built into the object.

The CCDData task should be able to properly deal with the
propogation of errors and propogate bad pixel frames
through each of the tasks.  It should also update the meta
data, units, and WCS information as the data are processed
through each step.

The following functions are required for performing basic CCD correction:
-creation of variance frame
-overscan subtraction
-bias subtraction
-trimming the data
-gain correction
-xtalk correction
-dark frames correction
-flat field correction
-illumination correction
-fringe correction
-scattered light correction
-cosmic ray cleaning
-distortion correction

In addition to the CCDData and CCDList class, the ccdproc does
require some additional features in order to properly
reduce CCD data. The following features are required
for basic processing of CCD data:
-fitting data
-combining data
-re-sampling data
-transforming data

All actions of ccdproc should be logged and recorded.

Multi-Extension FITS files can be handled by treating
each extension as a CCDData object and

'''

# ============
# Base Objects
# ============
'''
CCDData is an object that inherits from NDData class and specifically
describes an object created from the single readout of a CCD.

Users should be able to create a CCDData object from scratch, from
an existing NDData object, or a single extension from a FITS file.

In the case of the CCDData, the parameter 'uncertainty' will
be mapped to variance as that will be more explicitly describing
the information that will be kept for the processing of the

'''
data = 100 + 10 * np.random.random((110, 100))
# initializing without a unit raises an error
ccd = ccddata.CCDData(data=data) # ValueError

ccd = ccddata.CCDData(data=data, unit=u.adu)
ccd = ccddata.CCDData(NDData(data), unit=u.photon)

#Setting basic properties of the object
# ----------------------
ccddata.uncertainty = data**0.5
ccddata.mask = np.ones((110, 100), dtype=bool)
ccddata.flags = np.zeros((110, 100))
ccddata.wcs = None
ccddata.meta = {}
ccddata.header = {}  #header and meta are interchangable
ccddata.units = u.adu  # is this valid?


#The ccddata class should have a functional form to create a CCDData
#object directory from a fits file
ccd = ccddata.fits_ccddata_reader('img.fits', unit=u.adu)

# omitting a unit causes an error for now -- in the future an attempt should
# be made to extract the unit from the FITS header.
ccd = ccddata.fits_ccddata_reader('img.fits') # raises ValueError

# If a file has multiple extensions the desired hdu should be specified. It
# defaults to zero.
ccd = ccddata.fits_ccddata_reader('multi_extension.fits',
                                              image_unit=u.adu,
                                              hdu=2)

# any additional keywords are passed through to fits.open, with the exception
# of those related scaling (do_not_scale_image_data and scale_back)
ccd = ccddata.fits_ccddata_reader('multi_extension.fits',
                                              image_unit=u.adu,
                                              memmap=False)
# The call below raises a TypeErrror
ccd = ccddata.fits_ccddata_reader('multi_extension.fits',
                                              image_unit=u.adu,
                                              do_not_scale_image_data=True)


# This function should then be registered with astropy.io.registry, and the 
# FITS format auto-identified using the fits.connect.is_fits so
# the standard way for reading in a fits image will be
# the following: 
ccd = ccdproc.CCDData.read('img.fits', image_unit=u.adu)


# CCDData raises an error if no unit is provided; eventually an attempt to
# extract the unit from the FITS header should be made.  

# Writing is handled in a similar fashion; the image ccddata is written to
# the file img2.fits with:
ccdata.fits_ccddata_writer(ccd, 'img2.fits')

# all additional keywords are passed on to the underlying FITS writer, e.g.
ccddata.fits_ccddata_writer(ccd, 'img2.fits', clobber=True)

# The writer is registered with unified io so that in practice a user will do
ccd.write('img2.fits')

# NOTE: for now any flag, mask and/or unit for ccddata is discarded when
# writing. If you want all or some of that information preserved you must
# create the FITS files manually.

# To be completely explicit about not writing out addition information:
ccd.mask = np.ones(110, 100)
ccd.flags = np.zeros(110, 100)
ccd.write('img2.fits')

ccd2 = ccddata.CCDData.read('img2.fits', image_unit=u.adu)
assert ccd2.mask is None  # even though we set ccddata.mask before saving
assert ccd2.flag is None  # even though we set ccddata.flag before saving

# CCDData provides a convenience method to construct a FITS HDU from the data
# and metadata
hdu = ccd.to_hdu()

'''
Keyword is an object that represents a key, value pair for use in passing
data between functions in ``ccdproc``. The value is an astropy.units.Quantity,
with the unit specified explicitly when the Keyword instance is created.
The key is case-insensitive.
'''
key = ccdproc.Keyword('exposure', unit=u.s)
header = fits.Header()
header['exposure'] = 15.0
# value matched  by keyword name exposure
value = key.value_from(header)
assert value == 15 * u.s

# the value of a Keyword can also be set directly:
key.value = 20 * u.s

# String values are accommodated by not setting the unit and setting the value
# to a string

string_key = ccdproc.Keyword('filter')
string_key.value = 'V'

# Setting a string value when a unit has been specified is an error

bad_key = ccdproc.Keyword('exposure', unit=u.s)
bad_key.value = '30'  # raise a ValueError

''' Functional Requirements
 ----------------------
 A number of these different fucntions are convenient functions that
 just outline the process that is needed.   The important thing is that
 the process is being logged and that a clear process is being handled
 by each step to make building a pipeline easy.   Then again, it might
 not be easy to handle all possible steps which are needed, and the more
 important steps will be things which aren't already handled by NDData.

 All functions should propogate throught to the variance frame and
 bad pixel mask
'''


ccd.unit = u.adu
# The call below should raise an error because gain and readnoise are provided
# without units.  Gain and rdnoise should be Quantities
ccddata = ccdproc.create_variance(ccddata, gain=1.0, readnoise=5.0)

# The electron unit is provided by ccddata
gain = 1.0 * ccddata.electron / u.adu
readnoise = 5.0 * ccddata.electron
# This succeeds because the units are consistent
ccd = ccdproc.create_variance(ccd, gain=gain, readnoise=readnoise)

# with the gain units below there is a mismatch between the units of the
# image after scaling by the gain and the units of the readnoise...
gain = 1.0 *  u.photon / u.adu

# ...so this should fail with an error.
ccd = ccdproc.create_variance(ccd, gain=gain, readnoise=readnoise)


#Overscan subtract the data by providing the slice of ccd with
#the overscan section.  This will subtract off the median
#of each row
ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:,100:110])

#Overscan subtract the data by providing the slice of ccd with
#the overscan section.
ccd = ccdproc.subtract_overscan(ccd, fits_section='[101:110,:]')

#For the overscan region the astropy.model to fit to the data can
#be specified



#trim the images--the section gives the  part of the image to keep
#That the trim section is within the image.
ccd = ccdproc.trim_image(ccd[0:100,0:100])

#Using a section defined by fits convention
ccd = ccdproc.trim_image(ccd, fits_section='[1:100,1:100]')

#subtract the master bias. Although this is a convenience function as
#subtracting the two arrays will do the same thing. This should be able
#to handle logging of subtracting it off 
#Error checks: the masterbias and image are the same shape and units
masterbias = ccddata.CCDData(np.zeros((100, 100)), unit=u.adu)
ccd = ccdproc.subtract_bias(ccd, masterbias)

#correct for dark frames
#Options: Exposure time of the data image and the master dark image can be
#         specified as either an astropy.units.Quantity or as a ccdata.Keyword;
#         in the second case the exposure time will be extracted from the
#         metadata for each image.
masterdark = ccddata.CCDData(np.zeros((100, 100))+10, unit=u.adu)
masterdark.meta['exptime'] = 30.0
ccddata.meta['EXPOSURE'] = 15.0

exposure_time_key = ccdproc.Keyword('exposure',
                                    unit=u.s,
                                    synonyms=['exptime'])

# explicitly specify exposure times
ccd = ccdproc.subtract_dark(ccd, masterdark,
                                data_exposure=15 * u.s,
                                dark_exposure=30 * u.s,
                                scale=True
                                )

# get exposure times from metadata
ccd = ccdproc.subtract_dark(ccd, masterdark,
                                exposure_time=exposure_time_key,
                                scale=True)

#correct for gain--once again gain should have a unit and even an error
#associated with it.

# gain can be specified as a Quantity...
ccd = ccdproc.gain_correct(ccd, gain=1.0 * u.ph / u.adu)
# ...or the gain can be specified as a ccdproc.Keyword:
gain_key = ccdproc.Keyword('gain', unit=u.ph / u.adu)
ccd = ccdproc.gain_correct(ccd, gain=gain_key)

#Also the gain may be non-linear
#TODO: Not impliement in v0.1
ccd = ccdproc.gain_correct(ccd, gain=np.array([1.0, 0.5e-3]))

#although then this step should be apply before any other corrections
#if it is non-linear, but that is more up to the person processing their
#own data.

#crosstalk corrections--also potential a convenience function, but basically
#multiples the xtalkimage by the coeffient and then subtracts it.  It is kept
#general because this will be dependent on the CCD and the set up of the CCD.
#Not applicable for a single CCD situation
#Error checks: the xtalkimage and image are the same shape
#TODO: Not impliement in v0.1
xtalkimage = ccddata.CCDData(np.zeros((100, 100))+10, unit=u.adu)
ccd = ccdproc.xtalk_correct(ccddata, xtalkimage, coef=1e-3)

#flat field correction--this can either be a dome flat, sky flat, or an
#illumination corrected image.  This step should normalize by the value of the
#flatfield after dividing by it.
#Error checks: the  flatimage and image are the same shape
#Error checks: check for divive by zero
#Features: If the flat is less than minvalue, minvalue is used
flatimage = ccddata.CCDData(np.zeros((100, 100))+10, unit=u.adu)
ccd = ccdproc.flat_correct(ccd, flatimage, minvalue=1)

#fringe correction or any correction that requires subtracting
#off a potentially scaled image
#Error checks: the  flatimage and image are the same shape
#TODO: Not impliement in v0.1
fringeimage = ccddata.CCDData(np.zeros((100, 100))+10, unit=u.adu)
ccddata = ccdproc.fringe_correct(ccd, fringeimage, scale=1,
                                 operation='multiple')

#cosmic ray cleaning step--this should have options for different
#ways to do it with their associated steps.  We also might want to
#implement this as a slightly different step.  The cosmic ray cleaning
#step should update the mask and flags. So the user could have options
#to replace the cosmic rays, only flag the cosmic rays, or flag and
#mask the cosmic rays, or all of the above.
#median correct the cosmic rays
ccd = ccdproc.cosmicray_clean(ccd, thresh = 5, cr_func=ccdproc.cosmicray_median)

#remove cosmic rays following van dokkum method in LA COSMIC
ccddata = ccdproc.cosmicray_laplace(ccddata, thresh = 5, cr_func=ccdproc.cosmicray_lapace)

#Apply distortion corrections
#Either update the WCS or transform the frame
#TODO: Add after v0.1 when WCS is working
ccddata = ccdproc.distortion_correct(ccddata, distortion)

# =======
# Logging
# =======

# By logging we mean simply keeping track of what has been to each image in
# its as opposed to logging in the sense of the python logging module. Logging
# at that level is expected to be done by pipelines using the functions in
# ccdproc.

# for the purposes of illustration this document describes how logging would
# be handled for subtract_bias; handling for other functions would be similar.

# OPTION: One entry is added to the metadata for each processing step and the
# key added is the __name__ of the processing step.

# Subtracting bias like this:

ccddata = ccdproc.subtract_bias(ccddata, masterbias)

# adds a keyword to the metadata:

assert 'subtract_bias' in ccddata.meta  # name is the __name__ of the
                                        # processing step

# this allows fairly easy checking of whether the processing step is being
# repeated.

# OPTION: One entry is added to the metadata for each processing step and the
# key added is more human-friendly.

# Subtracting bias like this:

ccddata = ccdproc.subtract_bias(ccddata, masterbias)

# adds a keyword to the metadata:

assert 'bias_subtracted' in ccddata.meta  # name reads more naturally than
                                          # previous option

# OPTION: Each of the processing steps allows the user to specify a keyword
# that is added to the metadata. The keyword can either be a string or a
# ccdproc.Keyword instance

# add keyword as string:
ccddata = ccdproc.subtract_bias(ccddata, masterbias, add_keyword='SUBBIAS')

# add keyword/value using a ccdproc.Keyword object:
key = ccdproc.Keyword('calstat', unit=str)
key.value = 'B'
ccddata = ccdproc.subtract_bias(ccddata, masterbias,
                                add_keyword=key)

# =================
# Image combination
# =================

# The ``combine`` task from IRAF performs several functions:
# 1. Selection of images to be combined by image type with optional grouping
#    into subsets.
# 2. Offsetting of images based on either user-specified shifts or on WCS
#    information in the image metadata.
# 3. Rejection of pixels from inclusion in the combination based on masking,
#    threshold rejection prior to any image scaling or zero offsets, and
#    automatic rejection through a variety of algorithms (minmax, sigmaclip,
#    ccdclip, etc) that allow for scaling, zero offset and in some cases
#    weighting of the images being combined.
# 4. Scaling and/or zero offset of images before combining based on metadata
#    (e.g. image exposure) or image statistics (e.g image median, mode or
#    average determined by either an IRAF-selected subset of points or a
#    region of the image supplied by the user).
# 5. Combination of remaining pixels by either median or average.
# 6. Logging of the choices made by IRAF in carrying out the operation (e.g.
#    recording what zero offset was used for each image).

# As much as is practical, the ccdproc API separates these functions, discussed
# in detail below.

# 1. Image selection: this will not be provided by ccdproc (or at least not
#    considered part of image combination). We assume that the user will have
#    selected a set of images prior to beginning combination.

# 2. Position offsets: In ccdproc this is not part of combine. Instead,
#    offsets and other transforms are handled by ccdproc.transform, described
#    below under "Helper Function"

# The combination process begins with the creation of a Combiner object,
# initialized with a list of the images to be combined

from ccdproc import combiner
combine = combiner.Combiner([ccddata1, ccddata2, ccddata3])

#   automatic rejection by min/max, sigmaclip, ccdclip, etc. provided through
#   one method, with different helper functions

# min/max
combine.minmax_clipping(method=ccdproc.minmax, max_data=30000, data_min=-100)

# sigmaclip (relies on astropy.stats.funcs)
combine.sigma_clipping(low_thresh = 3.0, 
             high_thresh=3.0, 
             func=np.mean,
             dev_func=ma.std)
             
# TODO:  min/max pixels can be excluded in the sigma_clip. IRAF's clip excludes them
# Add exclude_extrema to min/max function

# ccdclip
# TODO: Not implemented in v0.1.  
#Can sigma_clipping be updated to use it?
combine.ccdclip_clipping( sigma_high=3.0, sigma_low=2.0,
             gain=1.0, read_noise=5.0,
             centerfunc=np.mean,
             exclude_extrema=True)

# 4. Image scaling/zero offset with scaling are set by the user prior
#    to passing the arrays to the task
#    TODO: Implement some scaling 

#    Image weights for use in the average combination can also be
#    set by the user
#    TODO: Implement some method for calculating the weighting
combine.weights = weights

# 5. The actual combination -- a couple of ways images can be combined

# median; the excluded pixels based on the individual image masks, threshold
# rejection, clipping, etc, are wrapped into a single mask for each image
combined_image = combine.median_combine()

# average; in this case image weights can also be specified if they 
# have been 
combined_image = combine.average_combine()

# ================
# Helper Functions
# ================

#fit a 1-D function with iterative rejections and the ability to
#select different functions to fit.
#other options are reject parameters, number of iteractions
#and/or convergernce limit
g = models.Gaussian1D(amplitude=1.2, mean=0.9, stddev=0.5)
coef = ccdproc.iterfit(x, y, model=g)


#fit a 2-D function with iterative rejections and the ability to
#select different functions to fit.
#other options are reject parameters, number of iteractions
#and/or convergernce limit
p = models.Polynomial2D(degree=2)
coef = ccdproc.iterfit(data, function=p)

#in addition to these operations, basic addition, subtraction
# multiplication, and division should work for CCDDATA objects
ccddata = ccddata + ccddata
ccddata2 = ccddata * 2
# TODO: This needs changes in base classes to work but 
# .add works for CCDDATA objects

#combine a set of NDData objects
alldata = ccdproc.combine([ccddata, ccddata2], method='average',
                          reject=None)

#re-sample the data to different binnings (either larger or smaller)
ccddata = ccdproc.rebin(ccddata, binning=(2, 2))

#tranform the data--ie shift, rotate, etc
#question--add convenience functions for image shifting and rotation?
#should udpate WCS although that would actually be the prefered method
ccddata = ccdproc.transform(ccddata, transform, conserve_flux=True)
