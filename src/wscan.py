#!/usr/bin/env python

# ---- Import standard modules to the python path.

from __future__ import division

import sys
import os
import glob
import random
import string
import shutil
import ConfigParser
import optparse
import json
import rlcompleter
import pdb
import operator

from panoptes_client import *

import pandas as pd
import numpy as np

from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline

from sqlalchemy.engine import create_engine

from jinja2 import Environment, FileSystemLoader

from matplotlib import use
use('agg')
from matplotlib import (pyplot as plt, cm)
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from gwpy.plotter import rcParams
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from glue import datafind

from gwpy.timeseries import TimeSeries
from gwpy.plotter import Plot
from gwpy.spectrogram import Spectrogram
from gwpy.segments import Segment

from pyomega import __version__
import pyomega.ML.make_pickle_for_linux as make_pickle
import pyomega.ML.labelling_test_glitches as label_glitches
import pyomega.API.projectStructure as Structure

pdb.Pdb.complete = rlcompleter.Completer(locals()).complete

###############################################################################
##########################                             ########################
##########################   Func: parse_commandline   ########################
##########################                             ########################
###############################################################################
# Definite Command line arguments here

def parse_commandline():
    """Parse the options given on the command-line.
    """
    parser = optparse.OptionParser()
    parser.add_option("--inifile", help="Name of ini file of params")
    parser.add_option("--eventTime", type=float,help="Trigger time of the glitch")
    parser.add_option("--outDir", help="Outdir of omega scan and omega scan webpage (i.e. your html directory)")
    parser.add_option("--pathToHTML", help="Outdir of omega scan and omega scan webpage (i.e. your html directory)")
    parser.add_option("--NSDF", action="store_true", default=False,help="No framecache file available want to use NSDF server")
    parser.add_option("--make-webpage", action="store_true", default=False,help="MAke output page")
    parser.add_option("--verbose", action="store_true", default=False,help="Run in Verbose Mode")
    opts, args = parser.parse_args()


    return opts


###############################################################################
##########################                     ################################
##########################      MAIN CODE      ################################
##########################                     ################################
###############################################################################

def main():
    # Parse commandline arguments

    opts = parse_commandline()

    ###########################################################################
    #                                   Parse Ini File                        #
    ###########################################################################

    # ---- Create configuration-file-parser object and read parameters file.
    cp = ConfigParser.ConfigParser()
    cp.read(opts.inifile)

    # ---- Read needed variables from [parameters] and [channels] sections.
    alwaysPlotFlag           = cp.getint('parameters','alwaysPlotFlag')
    sampleFrequency          = cp.getint('parameters','sampleFrequency')
    blockTime                = cp.getint('parameters','blockTime')
    searchFrequencyRange     = json.loads(cp.get('parameters','searchFrequencyRange'))
    searchQRange             = json.loads( cp.get('parameters','searchQRange'))
    searchMaximumEnergyLoss  = cp.getfloat('parameters','searchMaximumEnergyLoss')
    searchWindowDuration     = cp.getfloat('parameters','searchWindowDuration')
    whiteNoiseFalseRate      = cp.getfloat('parameters','whiteNoiseFalseRate')
    plotTimeRanges           = json.loads(cp.get('parameters','plotTimeRanges'))
    plotFrequencyRange       = json.loads(cp.get('parameters','plotFrequencyRange'))
    plotNormalizedERange     = json.loads(cp.get('parameters','plotNormalizedERange'))
    frameCacheFile           = cp.get('channels','frameCacheFile')
    frameTypes               = cp.get('channels','frameType').split(',')
    channelNames             = cp.get('channels','channelName').split(',')
    detectorName             = channelNames[0].split(':')[0]
    det                      = detectorName.split('1')[0]
 
    ###########################################################################
    #                           create output directory                       #
    ###########################################################################

    # if outputDirectory not specified, make one based on center time
    if opts.outDir is None:
        outDir = './scans'
    else:
        outDir = opts.outDir + '/'
    outDir += '/'

    # report status
    if not os.path.isdir(outDir):
        if opts.verbose:
            print('creating event directory')
        os.makedirs(outDir)
    if opts.verbose:
        print('outputDirectory:  {0}'.format(outDir))

    ########################################################################
    #     Determine if this is a normal omega scan or a Gravityspy         #
    #    omega scan with unique ID. If Gravity spy then additional         #
    #    files and what not must be generated                              #
    ########################################################################

    IDstring = "{0:.2f}".format(opts.eventTime)

    ###########################################################################
    #               Process Channel Data                                      #
    ###########################################################################

    # find closest sample time to event time
    centerTime = np.floor(opts.eventTime) +  np.round((opts.eventTime - np.floor(opts.eventTime)) * sampleFrequency) / sampleFrequency

    # determine segment start and stop times
    startTime = round(centerTime - blockTime / 2)
    stopTime = startTime + blockTime

    # This is for ordering the output page by SNR
    loudestEnergyAll = []
    channelNameAll   = []
    peakFreqAll      = []
    mostSignQAll     = []

    for channelName in channelNames:
        if 'STRAIN' in channelName:
            frameType = frameTypes[0]
        else:
            frameType = frameTypes[1]

        # Read in the data
        if opts.NSDF:
            data = TimeSeries.fetch(channelName,startTime,stopTime)
        else:
            connection = datafind.GWDataFindHTTPConnection()
            cache = connection.find_frame_urls(det, frameType, startTime, stopTime, urltype='file')
            data = TimeSeries.read(cache,channelName, format='gwf',start=startTime,end=stopTime)

        # resample data
        if data.sample_rate.decompose().value != sampleFrequency:
            data = data.resample(sampleFrequency)

	# Cropping the results before interpolation to save on time and memory
	# perform the q-transform
	try:
	    specsgrams = []
	    for iTimeWindow in plotTimeRanges:
		durForPlot = iTimeWindow/2
		try:
		    outseg = Segment(centerTime - durForPlot, centerTime + durForPlot)
		    qScan = data.q_transform(qrange=(4, 64), frange=(10, 2048),
				     gps=centerTime, search=0.5, tres=0.002,
				     fres=0.5, outseg=outseg, whiten=True)
                    qValue = qScan.q
		    qScan = qScan.crop(centerTime-iTimeWindow/2, centerTime+iTimeWindow/2)
		except:
		    outseg = Segment(centerTime - 2*durForPlot, centerTime + 2*durForPlot)
		    qScan = data.q_transform(qrange=(4, 64), frange=(10, 2048),
				     gps=centerTime, search=0.5, tres=0.002,
				     fres=0.5, outseg=outseg, whiten=True)
                    qValue = qScan.q
		    qScan = qScan.crop(centerTime-iTimeWindow/2, centerTime+iTimeWindow/2)
		specsgrams.append(qScan)

	    loudestEnergyAll.append(qScan.max().value)
	    peakFreqAll.append(qScan.yindex[np.where(qScan.value == qScan.max().value)[1]].value[0])
	    mostSignQAll.append(qValue)
	    channelNameAll.append(channelName)

	except:
	    print('bad channel {0}: skipping qScan'.format(channelName))
	    continue

	if opts.make_webpage:
	    # Set some plotting params
	    myfontsize = 15
	    mylabelfontsize = 20
	    myColor = 'k'
	    if detectorName == 'H1':
		title = "Hanford"
	    elif detectorName == 'L1':
		title = "Livingston"
	    else:
		title = "VIRGO"

	    if 1161907217 < startTime < 1164499217:
		title = title + ' - ER10'
	    elif startTime > 1164499217:
		title = title + ' - O2a'
	    elif 1126400000 < startTime < 1137250000:
		title = title + ' - O1'
	    else:
		raise ValueError("Time outside science or engineering run\
			   or more likely code not updated to reflect\
			   new science run")


	    # Create one image containing all spectogram grams
	    superFig = Plot(figsize=(27,6))
	    superFig.add_subplot(141, projection='timeseries')
	    superFig.add_subplot(142, projection='timeseries')
	    superFig.add_subplot(143, projection='timeseries')
	    superFig.add_subplot(144, projection='timeseries')
	    iN = 0

	    for iAx, spec in zip(superFig.axes, specsgrams):
		iAx.plot(spec)

		iAx.set_yscale('log', basey=2)
		iAx.set_xscale('linear')

		xticks = np.linspace(spec.xindex.min().value,spec.xindex.max().value,5)
		xticklabels = []
		dur = float(plotTimeRanges[iN])
		[xticklabels.append(str(i)) for i in np.linspace(-dur/2, dur/2, 5)]
		iAx.set_xticks(xticks)
		iAx.set_xticklabels(xticklabels)

		iAx.set_xlabel('Time (s)', labelpad=0.1, fontsize=mylabelfontsize, color=myColor)
		iAx.set_ylim(10, 2048)
		iAx.yaxis.set_major_formatter(ScalarFormatter())
		iAx.ticklabel_format(axis='y', style='plain')
		iN = iN + 1

		superFig.add_colorbar(ax=iAx, cmap='viridis', label='Normalized energy', clim=plotNormalizedERange, pad="3%", width="5%")

	    superFig.suptitle(title, fontsize=mylabelfontsize, color=myColor,x=0.51)
            superFig.save(outDir + channelName.replace(':','-') + '_' +IDstring + '_spectrogram_'  + '.png')

    if opts.make_webpage:

        channelNameAll   = [i.replace(':','-') for i in channelNameAll]
        loudestEnergyAll = [str(i) for i in loudestEnergyAll]
        peakFreqAll      = [str(i) for i in peakFreqAll]
        mostSignQAll     = [str(i) for i in mostSignQAll]

        # Zip SNR with channelName
        loudestEnergyAll = dict(zip(channelNameAll,loudestEnergyAll))
        peakFreqAll      = dict(zip(channelNameAll,peakFreqAll))
        mostSignQAll     = dict(zip(channelNameAll,mostSignQAll))

        plots = glob.glob(outDir + '*.png'.format(channelName))
        plots = [i.split('/')[-1] for i in plots]
        channelPlots = dict(zip(channelNameAll,plots))

        f1 = open(outDir + 'index.html','w')
        env = Environment(loader=FileSystemLoader('../'))
        template = env.get_template('webpage/omegatemplate.html')
        print >>f1, template.render(channelNames=channelNameAll, SNR=loudestEnergyAll,Q=mostSignQAll,FREQ=peakFreqAll,ID=IDstring,plots=channelPlots)
        f1.close()

        for channelName in channelNameAll:
            f2 = open(outDir + '%s.html' % channelName, 'w')
            template = env.get_template('webpage/channeltemplate.html'.format(opts.pathToHTML))
            # List plots for given channel
            print >>f2, template.render(channelNames=channelNameAll,thisChannel=channelName,plots=channelPlots)
            f2.close()

if __name__ == '__main__':
    main()
