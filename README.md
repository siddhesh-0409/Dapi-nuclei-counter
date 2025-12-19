# DAPI Nuclei Counter with Proliferation Analysis

## 📊 Overview
A production-ready Python tool for automated nuclei counting and proliferation analysis from fluorescence microscopy images. Processes **DAPI** (total nuclei) and **proliferation marker** channels (e.g., EdU/BrdU/Ki-67 in CY5, FITC, TRITC, etc.) to calculate proliferation percentages.

**Note**: The tool identifies proliferation marker images by the `_CY5` suffix in filenames, but this is configurable. If your proliferation marker is imaged in a different channel (e.g., FITC, TRITC, Alexa Fluor 488), you can easily adapt the code or rename your files to match.

## 🚀 Features
- **Multi-channel Analysis**: Processes DAPI (total nuclei) and any proliferation marker channel
- **Channel Flexibility**: Configured for CY5 by default, adaptable to FITC, TRITC, Alexa Fluor channels
- **Advanced Segmentation**: Uses watershed with peak_local_max for accurate nucleus separation
- **Batch Processing**: Automatically processes folders of image pairs
- **16-bit Support**: Preserves microscopy image quality and dynamic range
- **GUI Interface**: User-friendly Tkinter interface for easy operation
- **Comprehensive Outputs**: CSV measurements, diagnostic plots, proliferation percentages

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/siddhesh-0409/Dapi-nuclei-counter.git
cd Dapi-nuclei-counter

# Install dependencies
pip install -r requirements.txt
