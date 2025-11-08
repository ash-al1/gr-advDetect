#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-spectrumDetect author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Modifications:
# + Set gpuDevice to always be single gpu cuda device
# + Always use FP16 for speedup
# + Remove YOLO model & its parameters  [Maybe add back for baseline testing]
# + Adds documentation for classes & functions
# + Adds comments throughout code
# + Simpler if statements in some places
# + Count number of detections with detect_count instead of z+1

import numpy as np
import torchsig
import json
import cv2
import ultralytics
import torch
import torchaudio
import pmt
import time
import sigmf

from datetime import datetime
from gnuradio import gr
from math import pi, e
from ultralytics import YOLO
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
from .rxTime import rxTime                  # Hardware synchronized timestamps

class specDetect(gr.sync_block):
    """
    Real-time RF signal localization usign wideband detector

    Attributes:
        fc (float): Center frequency in Hz
        fs (float): Sample rate in Hz
        nfft (int): FFT size for spectrogram
        wb_model (Torch): Trained wideband detection model

    Methods:
        work(input_items, output_items): Main processing loop
        extract_narrowband_signal(): Downsample detected signals
    """
    def __init__(self, centerFrequency, sampleRate, vectorSize, nfft,
                 trainedWidebandModel, iou, conf, writeLabeledWBImages,
                 writeWBImages, writeWBIQFile, detectJson):

        # Define input/output Dtypes & size
        gr.sync_block.__init__(self,
            name="specDetect",
            in_sig=[(np.complex64,vectorSize)],
            out_sig=None)

        # Spectrum parameters
        self.detectJson = detectJson    
        self.fc = centerFrequency    
        self.fs = sampleRate            
        self.nfft = nfft
        self.nfftSamples = vectorSize

        # GPU model parameters
        self.gpuDevice = "cuda"
        self.wb_model_path = trainedWidebandModel 
        self.wb_model = YOLO(self.wb_model_path)

        # GPU
        torch.set_default_device(self.gpuDevice)
        self.wb_model.to(self.gpuDevice)

        # PMT output
        self.portName1 = "detect_pmt"
        self.message_port_register_out(pmt.intern(self.portName1))

        # Model hyperparameters
        self.iou = iou
        self.conf = conf

        # Timing & band
        self.d_rxTime = rxTime(vlen=nfft)
        self.use_PPS_time = False
        self.upper_limit = np.floor(self.fc + self.fs/2.0)
        self.lower_limit = np.floor(self.fc - self.fs/2.0)

        # Write to disk
        self.writeLabeledWBImages = writeLabeledWBImages
        self.writeWBImages = writeWBImages
        self.writeWBIQFile = writeWBIQFile


    def extract_narrowband_signal(self, input_data, r_cfreq,
        signal_freq, israte, osrate, truncate):
        """
        Frequency shift and resample complex IQ data down to baseband.

        Args:
            input_data (torch.Tensor): Complex IQ samples to process
            r_cfreq (float): Receiver Fc (Hz)
            signal_freq (float): Target signal Fc (Hz)
            israte (float): Input sample rate (Hz)
            osrate (float): Output sample rate (Hz)
            truncate (bool): Limit/pad output to 4096 samples

        Returns:
            tuple: (resampled_data, new_sample_rate, gpu_flag)
                - resampled_data (torch.Tensor): Downsampled IQ data
                - new_sample_rate (float): Sample rate (Hz)
        """

        with torch.cuda.device(self.gpuDevice):
            input_data = input_data.to(self.gpuDevice)
            x = torch.linspace(0, len(input_data)-1, len(input_data), dtype=torch.complex64, device=self.gpuDevice)
            fshift = (r_cfreq-signal_freq)/(israte*1.0)
            fv = e**(1j*2*pi*fshift*x)
            input_data = input_data * fv
            num_samples_in = len(input_data)
            num_samples = int(np.ceil(num_samples_in/(israte/osrate)))
            osrate_new = int((num_samples*israte)/num_samples_in)
            down_factor = int(israte/osrate_new)
            transform = torchaudio.transforms.Resample(int(israte/osrate_new),1,dtype=torch.float32).to(self.gpuDevice)
            if truncate:
                test = transform(input_data.real) + 1j*transform(input_data.imag)
                return test[:4096],israte/down_factor, True
            else:
                test = transform(input_data.real) + 1j*transform(input_data.imag)
                return test,israte/down_factor, True

                
    def work(self, input_items, output_items):
        """
        Main processing loop.

        Converts spectrogram to image format optionally writes these images with
        or without labels to disk, can also write IQ file to disk. Runs wideband
        model on vectorSize IQ and published results via PMT. Can write all
        predictions and IQ to disk using JSON and SigMF formats.

        Args:
            input_items (list): List containing input stream buffers.
                input_items[0] is array of complex64 vectors of size vectorSize
            output_items (list): List of output stream buffers (unused, no outputs)

        PMT Output:
            Dictionary with keys:
            - 'boxes_pmt': Dict of detections, each containing:
                - 'box_xyxy': YOLO bounding box format in pixel coordinates
                - 'center_freq': Signal center frequency (Hz)
                - 'bandwidth': Signal bandwidth (Hz)
                - 'wideband_modulation': YOLO classification
                - 'start_time': Signal start timestamp (ns since epoch)
                - 'duration': Signal duration (samples)
            - 'detect_count': Number of signals detected
            - 'detect': Boolean, True if any signals found
            - 'fcM': Center frequency (MHz)
            - 'fsM': Sample rate (MHz)
            - 'startTime': FFT window start timestamp (ns)
            - 'FFTSize': FFT size used
        """
        num_input_items = len(input_items[0])        
        in0 = input_items[0]
        nread = self.nitems_read(0)
        tags = self.get_tags_in_range(0, nread, nread + num_input_items)

        # Hardware timing (one initializion)
        if not self.use_PPS_time:
            for tag in tags:
                keyString = pmt.to_python(tag.key)
                if (keyString == 'rx_time'):
                    self.use_PPS_time = True
                    break

        # Get timestamp
        if (self.use_PPS_time):
            # Hardware
            self.d_rxTime.processTags(tags)
            current_time = self.d_rxTime.getNanoSecondsSinceEPOC(nread)
        else:
            # System
            current_time = np.uint64(time.time()*1e9)

        # Wait until timing is initialized
        if (not self.use_PPS_time or self.d_rxTime.isInitialized()):
            for inIdx in range(num_input_items):
                fcM = self.fc/1e6
                fsM = self.fs/1e6 

                # SigMF: write IQ to disk
                if self.writeWBIQFile:
                    dt = datetime.fromtimestamp(current_time // 1000000000)
                    in0[inIdx].tofile(f"{fcM}.{fsM}.{current_time}._cf32.sigmf-data")                    
                    metaWB = SigMFFile(
                        data_file=f"{fcM}.{fsM}.{current_time}._cf32.sigmf-data",
                        global_info = {
                            SigMFFile.DATATYPE_KEY: get_data_type_str(in0[inIdx]),
                            SigMFFile.SAMPLE_RATE_KEY: self.fs,
                            SigMFFile.DESCRIPTION_KEY: 'Complex F32 debug file.',
                        }
                    )
                    metaWB.add_capture(0, metadata={
                        SigMFFile.FREQUENCY_KEY: self.fc,
                        SigMFFile.DATETIME_KEY: dt.isoformat()+'Z',
                    })

                # IQ to GPU
                data = torch.from_numpy(in0[inIdx]).to(self.gpuDevice)
                
                # 1D magnitude spectrogram function
                spectrogram = torchaudio.transforms.Spectrogram( 
                              n_fft=self.nfft,
                              win_length=self.nfft,
                              hop_length=self.nfft,
                              window_fn=torch.blackman_window,
                              normalized=False,
                              center=False,
                              onesided=False,
                              power=2,
                              )
                spectrogram = spectrogram.to(self.gpuDevice).half()

                # Spectrogram maximum absolute (inf) normalization function
                norm = lambda x: torch.linalg.norm(
                              x,
                              ord=float("inf"),
                              keepdim=True,
                          )      

                # Normalize data
                x = spectrogram(data)
                x = x * (1 / norm(x.flatten()))
                x = torch.fft.fftshift(x,dim=0).flipud()
                x = 10*torch.log10(x+1e-12) 

                # RGB
                img_new = torch.zeros((self.nfft,self.nfft,3), device=self.gpuDevice)

                # Scale to [0, 1]
                a = torch.tensor([[1, torch.max(torch.max(x))], [1,torch.min(torch.min(x))]], device=self.gpuDevice)
                b = torch.tensor([1.0,0.0], device=self.gpuDevice)
                xx = torch.linalg.solve(a, b)
                intercept = xx[0]
                slope = xx[1]

                for j in range(3):
                    img_new[:,:,j] = (x*slope + intercept)

                # Reshape (1, 3, H, W)
                new_img_new = img_new.permute(-1,0,1).reshape(1,3,img_new.size(0),img_new.size(1)).to(self.gpuDevice)
                new_img_new = 1 - new_img_new                 

                # Run wideband detector
                result = self.wb_model(new_img_new, imgsz=self.nfft, iou=self.iou,
                                       conf=self.conf, half=True, verbose=False)
                plot_img = result[0].orig_img               
                detect_count = len(result[0].boxes.xyxy)

                # Write [labeled] image to disk 
                if self.writeWBImages:
                    cv2.imwrite(f"{fcM}.{fsM}.{current_time}.spectrogram.png",plot_img)
                if self.writeLabeledWBImages:
                    plot_img_l = result[0].plot()                
                    cv2.imwrite(f"{fcM}.{fsM}.{current_time}.spectrogram.labeled.png",plot_img_l)                        

                # Timestamp 
                startTime = np.uint64(current_time)
                endTime = np.uint64(startTime + int(((self.nfftSamples*1e9)/self.fs)))
                durationTime = endTime - startTime

                # PMT
                boxes_pmt = pmt.make_dict()
                boxes_pmt_dict = {}                
                detect_boxes = pmt.make_dict()
                detect_boxes_dict = {}                
                z = 0
                detect = False
                boxes_pmt_sum_dict = {}                

                # For each bbox
                for z, boxes_xyxy in enumerate(result[0].boxes.xyxy):
                    detect = True   
                    mod_nb = 'signal'
                    detect_dict = pmt.make_dict()
                    detectDict = {}   
                    
                    # Classification from wideband model
                    mod_wb = result[0].names[int(result[0].boxes.cls[z].cpu().numpy())]  
                    mod_wb = 'signal'       

                    # Bbox coordinates
                    box_xyxy = boxes_xyxy.cpu().numpy()
                    box_xywh = result[0].boxes.xywh[z].cpu().numpy()

                    # Pixel coordinates to frequency & time
                    center_freq = ((float(self.fs)/2.0)-(float(box_xywh[1]/self.nfft)*float(self.fs))+self.fc)
                    top_freq = ((float(self.fs)/2.0)-((box_xyxy[1]/self.nfft)*float(self.fs))+self.fc)
                    bottom_freq = ((float(self.fs)/2.0)-((box_xyxy[3]/self.nfft)*float(self.fs))+self.fc)
                    bandwidth = top_freq - bottom_freq

                    # BBox Duration
                    start_sample = int(box_xyxy[0])*int(self.nfft)
                    end_sample = int(box_xyxy[2])*int(self.nfft)
                    duration = end_sample - start_sample                  

                    # Timestamp bbox (start_sample may be 0)
                    offset = np.uint64((start_sample * 1e9) / self.fs)
                    length = np.uint64((duration * 1e9) / self.fs)
                    start_time = current_time + offset
                    end_time = start_time + length

                    # Annotation to SigMF
                    if self.writeWBIQFile:
                        metaWB.add_annotation(start_sample, duration, metadata = {
                            SigMFFile.FLO_KEY: bottom_freq,
                            SigMFFile.FHI_KEY: top_freq,
                            SigMFFile.COMMENT_KEY: 'wb modulation: '+mod_wb+' nb modulation: '+mod_nb,
                        })                  

                    # PMT message dictionary
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('box_xyxy'),pmt.to_pmt(box_xyxy))   
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('box_xywh'),pmt.to_pmt(box_xywh)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('narrowband_modulation'),pmt.to_pmt(mod_nb))
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('wideband_modulation'),pmt.to_pmt(mod_wb)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('center_freq'),pmt.from_float(center_freq))
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('top_freq'),pmt.from_float(top_freq)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('bottom_freq'),pmt.from_float(bottom_freq)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('bandwidth'),pmt.from_float(bandwidth)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('start_sample'),pmt.from_long(start_sample)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('end_sample'),pmt.from_long(end_sample))  
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('duration'),pmt.from_long(duration)) 
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('start_time'),pmt.from_uint64(start_time))      
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('end_time'),pmt.from_uint64(end_time))  
                    detect_dict = pmt.dict_add(detect_dict, pmt.intern('length'),pmt.from_uint64(length))           
                    boxes_pmt = pmt.dict_add(boxes_pmt, pmt.intern(str(z)),detect_dict)

                    # JSON dictionary
                    if self.detectJson:
                        detectDict['box_xyxy'] = [float(i) for i in list(box_xyxy)] 
                        detectDict['box_xywh'] = [float(i) for i in list(box_xywh)]
                        detectDict['center_freq'] = float(center_freq)
                        detectDict['modulation_wb'] = mod_wb
                        detectDict['modulation_nb'] = mod_nb
                        detectDict['top_freq'] = float(top_freq) 
                        detectDict['bottom_freq'] = float(bottom_freq)
                        detectDict['bandwidth'] = float(bandwidth)
                        detectDict['start_sample'] = int(start_sample)
                        detectDict['end_sample'] = int(end_sample)
                        detectDict['duration'] = int(duration)
                        detectDict['start_time'] = int(start_time)
                        detectDict['end_time'] = int(end_time) 
                        detectDict['length'] = int(length) 
                        boxes_pmt_sum_dict[str(z)] = detectDict 
                
                # Write JSON detection
                if self.detectJson:                
                    boxes_pmt_dict['detects'] = boxes_pmt_sum_dict
                    boxes_pmt_dict['detect_count'] = int(detect_count)
                    boxes_pmt_dict['detect'] = detect
                    detect_boxes_dict['boxes_pmt'] = boxes_pmt_dict  
                    detect_boxes_dict['fcM'] = float(fcM) 
                    detect_boxes_dict['fsM'] = float(fsM) 
                    detect_boxes_dict['startTime'] = int(startTime)      
                    detect_boxes_dict['endTime'] = int(endTime) 
                    detect_boxes_dict['durationTime'] = int(durationTime)       
                    detect_boxes_dict['FFTSize'] = int(self.nfft)      
                    detect_boxes_dict['FFTxFFT'] = int(self.nfftSamples)
                    detect_boxes_dict['trainedWidebandModel'] = str(self.wb_model_path)
                    detect_boxes_dict['half'] = str("True")
                    detect_boxes_dict['iou'] = str(self.iou)
                    detect_boxes_dict['conf'] = str(self.conf)
                    with open(f"{fcM}.{fsM}.{current_time}.detect.json", "w",encoding='utf-8') as outfile:
                        json.dump(detect_boxes_dict, outfile)                   

                # Finalize PMT message
                boxes_pmt = pmt.dict_add(boxes_pmt, pmt.intern('detect_count'), pmt.from_long(detect_count))
                boxes_pmt = pmt.dict_add(boxes_pmt, pmt.intern('detect'),pmt.from_bool(detect))  
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('boxes_pmt'),boxes_pmt)
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('plot_img'),pmt.to_pmt(plot_img))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('fcM'),pmt.from_float(fcM))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('fsM'),pmt.from_float(fsM))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('startTime'),pmt.from_uint64(startTime))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('endTime'),pmt.from_uint64(endTime))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('durationTime'),pmt.from_uint64(durationTime))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('FFTSize'),pmt.from_long(int(self.nfft)))
                detect_boxes = pmt.dict_add(detect_boxes, pmt.intern('FFTxFFT'),pmt.from_long(int(self.nfftSamples)))

                # Publish PMT message
                self.message_port_pub(pmt.intern(self.portName1), detect_boxes)
                if self.writeWBIQFile:
                    metaWB.tofile(f"{fcM}.{fsM}.{current_time}._cf32.sigmf-data")
 
        return num_input_items
