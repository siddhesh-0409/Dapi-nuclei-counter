# DAPI Nuclei Counter with Proliferation Analysis

A production-ready Python tool for automated nuclei counting and proliferation analysis from fluorescence microscopy images.

## Features

- **DAPI/CY5 Analysis**: Processes both total nuclei (DAPI) and proliferating cells (CY5)
- **Advanced Segmentation**: Uses watershed with peak_local_max for accurate separation
- **Batch Processing**: Automatically processes folders of image pairs
- **16-bit Support**: Preserves microscopy image quality
- **Comprehensive Outputs**: CSV measurements, diagnostic plots, proliferation percentages
- **GUI Interface**: User-friendly Tkinter interface

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dapi-nuclei-counter.git
cd dapi-nuclei-counter

# Install dependencies
pip install -r requirements.txt