#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 gr-spectrumDetect author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Modifications:
# + Commented timing information out when plotting spectrogram
# + Comments & documentation throughout


import numpy as np
import pmt

from gnuradio import gr
from PyQt5 import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt
from matplotlib.path import Path 
from matplotlib.patches import PathPatch

class spectrumPlot(gr.sync_block, QWidget):
    """
    Real-time spectrogram visualization

    Displays spectrograms with bounding boxes around detected signals,
    annotated with frequency, timing, and modulation information.
    Receives detection data via PMT messages from specDetect block.

    Attributes:
        save (bool): Save each plot as PNG
        lbl (str): Title
        figure (matplotlib.figure.Figure): Main plot
        ax (matplotlib.axes.Axes): Plot axes
        canvas (FigureCanvas): Qt5 canvas
    """
    def __init__(self,save,lbl):
        gr.sync_block.__init__(self,
            name="spectrumPlot",
            in_sig=None,
            out_sig=None)       

        QWidget.__init__(self)   

        # Plot configuration
        self.save = save       
        self.figure = plt.figure(figsize=(40,40))
        self.ax = plt.subplot()
        self.lbl = lbl
        self.figure.suptitle(self.lbl, fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Time Bins")
        self.ax.set_ylabel("FFT Bins")

        # QT canvas
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Codes for bbox
        self.codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]
        
        # PMT input port
        self.portName1 = "detect_pmt"
        self.message_port_register_in(pmt.intern(self.portName1))
        self.set_msg_handler(pmt.intern(self.portName1), self.plot)
       

    def plot(self, msg):
        """
        Plot spectrogram with detection overlays from PMT message.

        Args:
            msg (pmt): PMT dictionary containing:
                - plot_img: Spectrogram image array
                - boxes_pmt: Detection bounding boxes and metadata
                - fcM: Center frequency (MHz)
                - fsM: Sample rate (MHz)
                - startTime/endTime: Timestamps (ns)
                - FFTSize: FFT dimensions
        """
        self.ax.cla()

        # Unpack PMT message
        detect_boxes = pmt.to_python(msg)
        nfft = detect_boxes['FFTSize'] 
        plot_img = detect_boxes['plot_img'].reshape(nfft,nfft,3)
        cfreqMHz = detect_boxes['fcM']
        plotFreqBW = detect_boxes['fsM']
        duration = str(detect_boxes['durationTime'])
        FFTSize = detect_boxes['FFTSize']
        nfftSamples = detect_boxes['FFTxFFT']

        # Draw box around signal of interest
        if detect_boxes['boxes_pmt']['detect']:
            for cnt in range(detect_boxes['boxes_pmt']['detect_count']):
                detection = detect_boxes['boxes_pmt'][str(cnt)]
                plot_box_xyxy = detection['box_xyxy']  # [x1, y1, x2, y2]
                plot_box_xywh = detection['box_xywh']  # [x_c, y_c, w, h]

                # Bbox vertices
                verts = [
                    (plot_box_xyxy[0], plot_box_xyxy[1] + plot_box_xywh[3]),
                    (plot_box_xyxy[0], plot_box_xyxy[1]),
                    (plot_box_xyxy[0] + plot_box_xywh[2], plot_box_xyxy[1]),
                    (plot_box_xyxy[2], plot_box_xyxy[3]),
                    (0., 0.),
                ]
                
                # Draw bbox
                path = Path(verts, self.codes)
                patch = PathPatch(path, facecolor='none', lw=2, edgecolor='red', alpha=.3)
                self.ax.add_patch(patch)

                # Timing annotation
                #self.ax.text(plot_box_xyxy[0], plot_box_xyxy[1],
                #            f"st: {detection['start_time']}(ns)",
                #            color='cyan', fontsize=10)
                
                # Frequency annotation
                self.ax.text(plot_box_xyxy[0], plot_box_xyxy[1] + (0.5 * plot_box_xywh[3]),
                            f"fc: {detection['center_freq']/1e6:.6}(MHz)",
                            color='yellow', fontsize=10)

                # Wideband classification result (if present)
                if detection['wideband_modulation'] != 'signal':
                    self.ax.text(plot_box_xyxy[0], plot_box_xyxy[1] + (0.7 * plot_box_xywh[3]), 
                                f"wb_mod: {detection['wideband_modulation']}", 
                                color='chartreuse', fontsize=10)

                # Narrowband classification result (if present)
                if detection['narrowband_modulation'] != 'signal':
                    self.ax.text(plot_box_xyxy[0], plot_box_xyxy[1] + (0.9 * plot_box_xywh[3]),
                                f"nb_mod: {detection['narrowband_modulation']}",
                                color='lawngreen', fontsize=10)

        self.ax.imshow(plot_img)

        # Axis
        #self.ax.set_xlabel(
        #    f"Duration: {duration}(ns), Start Time (ns): {detect_boxes['startTime']} "
        #    f"End Time (ns): {detect_boxes['endTime']}"
        #)
        self.ax.set_ylabel(
            f"Bandwidth: {plotFreqBW}(MHz), Center Frequency: {cfreqMHz}(MHz)"
        )

        # Set tick labels for time and frequency
        mid_time = np.uint64(detect_boxes['startTime']) + np.uint64(detect_boxes['durationTime'] / 2)
        self.ax.set_xticks(
            [0, (nfft / 2) - 1, nfft - 1],
            [detect_boxes['startTime'], mid_time, detect_boxes['endTime']]
        )
        
        self.ax.set_yticks(
            [0, (nfft / 2) - 1, nfft - 1],
            [cfreqMHz + plotFreqBW / 2.0, cfreqMHz, cfreqMHz - plotFreqBW / 2.0]
        )

        # Save to disk
        if self.save:
            self.figure.savefig(str(detect_boxes['startTime'])+'.png')

        self.canvas.draw()
