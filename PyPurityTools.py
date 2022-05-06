from struct import unpack, calcsize
import numpy as np
from scipy.signal import butter,filtfilt,lfilter,lfilter_zi






class PyPurityTools:
    
    #Function to unpack the header
    @staticmethod
    def unpackHeader(header,params_pattern='=IBdddd'):
        sHeader = unpack(params_pattern,header)
        numSamples=sHeader[0]
        bytesPerSample=sHeader[1]
        v_off=sHeader[2]
        v_scale=sHeader[3]
        h_off=sHeader[4]
        h_scale=sHeader[5]
        return numSamples,bytesPerSample,v_off,v_scale,h_off,h_scale

    @staticmethod
    def getScopeWaveforms(waveFilename,doBaseline=False):
        #params_pattern defines the structure we want to read in 
        params_pattern = '=IBdddd' # (num_samples, sample_bytes, v_off, v_scale, h_off, h_scale, [samples]) ...
        struct_size = calcsize(params_pattern)
        #print(struct_size)

        with open(waveFilename,"rb") as fWave:
            waveList=[]
            while True:
                #First read and unpack the headers
                header = fWave.read(struct_size)
                if not header: break
                numSamples,bytesPerSample,v_off,v_scale,h_off,h_scale=PyPurityTools.unpackHeader(header,params_pattern) 
    
        
                #Now read in the waveform samples
                dataType=np.dtype('>i1')
                dataList=np.fromfile(fWave,dataType,numSamples)

                #Convert to volts and seconds and subtract a baseline
                voltList=dataList*v_scale
                if doBaseline:
                    baseline=np.mean(voltList[100:8000])
                    voltList-=baseline
                else:
                    voltList-=v_off
                waveList.append(voltList)
        
        waveList=np.vstack(waveList)
        #Make an array of time values (assumimg they are all the same sample rates in the file)
        sampList=np.arange(numSamples)
        timeList=sampList*h_scale
        timeList-=h_off
        timeList*=1e6 #Convert to microseconds
        return waveList,timeList

    #Convert to dB... relative to 1.0
    def convertTodB(cVals):
        N=len(cVals)
        return 10*np.log10(2.0/N * np.abs(cVals[:N//2]))
        
    #Wrapper for a lowpass butterworth filter
    #Not very efficient at the moment
    def butter_lowpass_filter(data, cutoff, fs, order):
        # Filter requirements.
        #T = 5.0         # Sample Period
        #fs = 30.0       # sample rate, Hz
        #cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        #order = 2       # sin wave can be approx represented as quadratic
        #n = int(T * fs) # total number of samples
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
     #Wrapper for a lowpass butterworth filter
    #Not very efficient at the moment
    def butter_lowpass_lfilter(data, cutoff, fs, order):
        # Filter requirements.
        #T = 5.0         # Sample Period
        #fs = 30.0       # sample rate, Hz
        #cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        #order = 2       # sin wave can be approx represented as quadratic
        #n = int(T * fs) # total number of samples
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        zi = lfilter_zi(b, a)
        z, _ = lfilter(b, a, data, zi=zi*data[0])
        return z
    