#import spcdal
import os
import spectral
from spectral import envi
from spectral import resampling
import numpy as np
import matplotlib.pyplot as plt
import lazypredict
import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import itertools
import matplotlib.pyplot as plt



class txt2sli:
    def __init__(self, dir = 'Ecostress_Veg_Data/eco_veg/'):
        self.dir = dir
        self.create_metadata_keys()
        self.read_data()
        self.save_spectral_libraries()
    
    def create_metadata_keys(self):
        self.metadata_keys = [
            'Full Name',
            'Sensor',
            'Name',
            'Type',
            'Class',
            'Subclass',
            'Particle Size',
            'Sample No.',
            'Owner',
            'Wavelength Range:',
            'Origin',
            'Collection Date',
            'Description',
            'Measurement',
            'First Column',
            'Second Column',
            'X Units',
            'Y Units',
            'First X Value',
            'Last X Value',
            'Number of X Values',
            'Additional Information',
            'Class Metadata'
            'Name',
            'Type',
            'Class',
            'Genus',
            'Species',
            'Sample No.',
            'Owner',
            'Chemistry',
            'Biophysical Properties',
            'Sampling Notes',
            'Citation',
            'Particle Size',
            'Sample No.',
            'Owner',
            'Wavelength Range',
            'Origin',
            'Collection Date',
            'Description',
            'Mineral',
            'File Name',
            'XRD Analysis',
            'Spectrum',
            'Wl'
        ]

    def read_data_from_txt_file(self, fname, data):
        my_file = open(fname, "r", errors="ignore") 
        data_list = my_file.read().split('\n')
        my_file.close() 
        
        reading_header = True
        #wl = []
        #spec = []
        for row in data_list:
            row = row.strip()
            if len(row)>0:
                
                if reading_header:
                    #print('reading header')
                    # If reading the header portion of the file
                    idx = row.find(':')
                    data[row[:idx]] = row[(idx+1):].strip()
                    if row[:idx]=='Additional Information':
                        reading_header = False
                        wl = []
                        spec = []
                    
                else:
                    #print('reading data')
                    #print(row)
                    # If reading the spectrum and wavelength portion of the file
                    try:
                        wl.append(float(row[:row.find('.')+5]))  
                        spec.append(float(row[(row.rfind('.')-2):]))
                    except:
                        wl.append(float(row[(row.find('\t')+2):]))                     
                        spec.append(float(row[row.find('\t'):])) 

        if reading_header == False:
            data['Spectrum'] = np.asarray(spec).flatten()
            data['Wl'] = np.asarray(wl).flatten()
            
        return data


    def read_data(self):
        dir_list = os.listdir(self.dir)
        sensors = []
        data = {}
        for fname in dir_list:
            data_keys = list(data.keys())
            
            if 'spectrum' in fname:
                fullname_sensor = fname[:(fname.find('spectrum')-1)]
                full_name = fullname_sensor[:(fullname_sensor.rfind('.'))]
                if fullname_sensor not in data_keys:
                    data[fullname_sensor] = dict.fromkeys(self.metadata_keys, None)  
                data[fullname_sensor]['Full Name'] = full_name
                data[fullname_sensor] = self.read_data_from_txt_file(os.path.join(self.dir,fname), data[fullname_sensor])
                wl_start = np.min([float(data[fullname_sensor]['First X Value']),float(data[fullname_sensor]['Last X Value'])])
                wl_end = np.max([float(data[fullname_sensor]['First X Value']),float(data[fullname_sensor]['Last X Value'])])
                sensor = (fullname_sensor[(fullname_sensor.rfind('.')+1):]+
                        '_'+str(data[fullname_sensor]['Number of X Values'])+
                        '_'+str(wl_start)[:4]+
                        '_'+str(wl_end)[:4])
                data[fullname_sensor]['Sensor'] = sensor
                sensors.append(sensor)
                
            if 'ancillary' in fname:
                fullname_sensor = fname[:(fname.find('ancillary')-1)]
                full_name = fullname_sensor[:(fullname_sensor.rfind('.'))]
                sensors.append(sensor)
                if fullname_sensor not in data_keys:
                    data[fullname_sensor] = dict.fromkeys(self.metadata_keys, None)  
                data[fullname_sensor]['Full Name'] = full_name
                data[fullname_sensor] = self.read_data_from_txt_file(os.path.join(self.dir,fname), data[fullname_sensor])

        sensors = np.unique(sensors)
        self.sensors = sensors
        self.data = data

    def save_spectral_libraries(self):
        # convert to spectral libraries and save files
        data_keys = list(self.data.keys())
        libs = {}

        #Iterate through the sensors
        count = 0
        for sensor in self.sensors:
            fname_lib = 'ecostrs_'+sensor
            read_wl = True
            
            # build the header
            header = {}
            header['spectra names'] = []
            header['chemistry_water'] = []
            header['chemistry_nitrogen'] = []
            header['chemistry_carbon'] = []
            
            # Iterate through the spectra
            for k in data_keys:
                d = self.data[k]
                # Check if this spectra was collected using current sensor
                if d['Sensor']==sensor:            
                    count = count + 1   
                    # get the sensor info (wavelengths and possibly fwhm)
                    if read_wl:
                        header['wavelength'] = list(d['Wl'])
                        read_wl = False

                    water, nitrogen, carbon = '-1', '-1', '-1'
                    
                    if d['Chemistry'] is not None:
                        original_array = d['Chemistry']
                        
                        if 'Water Content' in original_array:
                            # find water content
                            start_idx = original_array.find('Water Content') + 14
                            end_idx = original_array[start_idx:].find('%')
                            water = original_array[start_idx:(end_idx + start_idx)]
                            
                        if 'Nitrogen' in original_array:
                            # find water content
                            start_idx = original_array.find('Nitrogen') + 9
                            end_idx = original_array[start_idx:].find('%')
                            #print('start_end_index:',original_array[start_idx:(end_idx + start_idx)])
                            nitrogen = original_array[start_idx:(end_idx + start_idx)]
                            
                        if 'Carbon' in original_array:
                            # find water content
                            start_idx = original_array.find('Carbon') + 7
                            end_idx = original_array[start_idx:].find('%')
                            #print('start_end_index:',original_array[start_idx:(end_idx + start_idx)])
                            carbon = original_array[start_idx:(end_idx + start_idx)]

                    header['chemistry_water'].append(water)
                    header['chemistry_nitrogen'].append(nitrogen)
                    header['chemistry_carbon'].append(carbon)
                    #print('Header_Chemistry: ',header['chemistry_water'])  
                    
                    # add the spectra name and values  
                    header['spectra names'].append(d['Full Name'])
                    # add the data for the spectrum
                    if len(header['spectra names'])==1:
                        spectra_arr = d['Spectrum']
                    else:
                        spectra_arr = np.vstack((spectra_arr,d['Spectrum']))  
            
            if count > 0:      
                # save the collection as an ENVI spectral library
                lib = envi.SpectralLibrary(spectra_arr, header, [])
                lib.save(fname_lib)

                print(fname_lib+'   Number of Spectra: '+str(len(header['spectra names'])))
                count = count + len(header['spectra names'])
                
        print('Total Number of Spectra in All Libraries: '+str(count))

