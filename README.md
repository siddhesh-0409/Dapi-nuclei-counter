# DAPI Stained Nuclei Counter with Proliferation Analysis

## 📊 Overview
A production-ready Python tool for automated nuclei counting and proliferation analysis from fluorescence microscopy images. Processes DAPI and CY5 channel images to count total nuclei and proliferating cells.

## 🚀 Features
- **DAPI/CY5 Analysis**: Processes both total nuclei (DAPI) and proliferating cells (CY5)
- **Advanced Segmentation**: Uses watershed with peak_local_max for accurate separation
- **Batch Processing**: Automatically processes folders of image pairs
- **16-bit Support**: Preserves microscopy image quality
- **GUI Interface**: User-friendly Tkinter interface
- **Comprehensive Outputs**: CSV measurements, diagnostic plots, proliferation percentages

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/siddhesh-0409/Dapi-nuclei-counter.git
cd Dapi-nuclei-counter

# Install dependencies
pip install -r requirements.txt
