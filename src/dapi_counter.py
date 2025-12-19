import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
import tkinter as tk
from tkinter import filedialog, messagebox
# Added ttk for the progress bar
from tkinter import ttk
import traceback
import logging
import os
import json
from datetime import datetime
import threading # To run GUI updates smoothly
import re # For sorting filenames numerically

# Scikit-image imports for the full pipeline
from skimage import io, draw, util
from skimage import filters, measure, morphology, segmentation, restoration, feature
from skimage.color import label2rgb
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral

# --- Class Definition (Unchanged) ---
class MaximumAccuracyDAPICounter:
    """
    Production-ready DAPI nuclei counter with maximum accuracy configuration.
    Processes _DAPI and _CY5 files, preserving 16-bit precision.
    Uses peak_local_max for watershed markers.
    """

    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()
        self.setup_logging()

    def _get_default_config(self) -> dict:
        """Return comprehensive default configuration for maximum accuracy"""
        # Default config remains the same
        return {
            'processing': {
                'sigma_spatial': 3,
                'sigma_color_ratio': 0.1,
                'otsu_classes': 2,
                'min_distance': 10 # Crucial watershed parameter
            },
            'filtering': {
                'size_min': 200,
                'size_max': 25000,
                'circularity_min': 0.40,
                'circularity_max': 1.00,
                'solidity_min': 0.85,
                'eccentricity_max': 0.9,
                'intensity_min_ratio': 0.1 # Default ratio
            },
            'visualization': {
                'generate_all_plots': True,
            }
        }

    def setup_logging(self):
        """Setup comprehensive logging for production use"""
        # Logging setup remains the same
        try:
            if os.name == 'nt':
                base_dir = Path(os.environ.get('USERPROFILE', '.'))
            else:
                base_dir = Path(os.environ.get('HOME', '.'))

            log_dir = base_dir / "dapi_analysis_logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"dapi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                handler.close()

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Logging initialized. Log file: {log_file}")

        except Exception as e:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"File logging failed, using console only: {str(e)}")

    def load_image_16bit(self, image_path: str) -> np.ndarray:
        """Loads image, ensures 16-bit, extracts channel if needed."""
        # load_image_16bit remains the same
        try:
            img = io.imread(image_path)
        except FileNotFoundError:
            self.logger.error(f"File not found at {image_path}")
            raise

        if img.ndim >= 3:
            img = img[..., 0]
            self.logger.info("Multi-channel image detected. Extracted channel 0.")

        if img.dtype == np.uint8:
            self.logger.warning("8-bit image detected. Upscaling to 16-bit range (0-65535).")
            img = (img.astype(np.float32) / 255.0 * 65535).astype(np.uint16)

        if not np.issubdtype(img.dtype, np.integer):
            if img.min() >= 0 and img.max() <= 1:
                 self.logger.info(f"Float image detected (range {img.min()}-{img.max()}). Assuming [0,1], scaling to uint16.")
                 img = util.img_as_uint(img, force_copy=True)
            else:
                 self.logger.warning(f"Float image detected with unexpected range ({img.min()}-{img.max()}). Attempting direct conversion to uint16.")
                 img = img.astype(np.uint16, copy=False)

        self.logger.info(f"Loaded image: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
        return img

    def advanced_denoise(self, image: np.ndarray) -> np.ndarray:
        """Applies Bilateral filter preserving 16-bit range."""
        # advanced_denoise remains the same
        sigma_spatial = self.config['processing']['sigma_spatial']
        sigma_color_ratio = self.config['processing']['sigma_color_ratio']

        img_range = image.max() - image.min()
        sigma_color = float(img_range) * sigma_color_ratio if img_range > 0 else 1.0

        self.logger.info(f"Applying Bilateral Filter (sigma_spatial={sigma_spatial}, sigma_color={sigma_color:.0f})")

        denoised = denoise_bilateral(
            image,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
            channel_axis=None
        )
        if not np.issubdtype(denoised.dtype, np.integer):
             denoised = util.img_as_uint(denoised, force_copy=True)
        return denoised

    def multi_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """Applies Multi-Otsu thresholding."""
        # multi_otsu_threshold remains the same
        classes = self.config['processing']['otsu_classes']
        self.logger.info(f"Applying Multi-Otsu threshold (classes={classes})")

        try:
            if image.min() == image.max():
                 self.logger.warning("Image is constant. Using standard Otsu (might return empty mask).")
                 raise ValueError("Image is constant")
            thresholds = filters.threshold_multiotsu(image, classes=classes)
            binary = image > thresholds[-1]
            self.logger.info(f"Multi-Otsu threshold value: {thresholds[-1]}")
        except Exception as e:
            self.logger.warning(f"Multi-Otsu failed ({e}). Falling back to standard Otsu.")
            try:
                 thresh = filters.threshold_otsu(image)
                 binary = image > thresh
                 self.logger.info(f"Fallback Otsu threshold: {thresh}")
            except Exception as otsu_e:
                 self.logger.error(f"Standard Otsu also failed: {otsu_e}. Returning empty mask.")
                 binary = np.zeros_like(image, dtype=bool)

        return binary

    def enhanced_watershed(self, binary_mask: np.ndarray) -> tuple:
        """Applies marker-controlled watershed using peak_local_max."""
        # enhanced_watershed remains the same
        self.logger.info("Performing enhanced watershed segmentation using peak_local_max...")

        min_distance = self.config['processing']['min_distance']

        distance = ndimage.distance_transform_edt(binary_mask)
        distance_smooth = distance # Use unsmoothed for peaks

        coords = feature.peak_local_max(
            distance_smooth,
            min_distance=min_distance,
            threshold_rel=0.1,
            labels=binary_mask
        )
        markers_mask = np.zeros(distance.shape, dtype=bool)
        if coords.size > 0:
             markers_mask[tuple(coords.T)] = True
        markers = measure.label(markers_mask)

        n_markers = markers.max()
        self.logger.info(f"Found {n_markers} markers using peak_local_max (min_distance={min_distance})")

        if n_markers == 0:
            self.logger.warning("No markers found for watershed! Labeling connected components instead.")
            labels = measure.label(binary_mask)
            distance_for_watershed = distance
        else:
            distance_for_watershed = ndimage.gaussian_filter(distance, sigma=1.0)
            labels = segmentation.watershed(
                -distance_for_watershed,
                markers,
                mask=binary_mask,
                compactness=0.1,
                watershed_line=True
            )

        self.logger.info(f"Segmentation complete. Found {np.max(labels)} regions.")
        return labels, distance_for_watershed


    def measure_and_filter_nuclei(self, labels: np.ndarray, original_image: np.ndarray) -> tuple:
        """Combines filtering and measurement on 16-bit original."""
        # measure_and_filter_nuclei remains the same
        self.logger.info("Filtering objects and measuring 16-bit properties...")

        filt_cfg = self.config['filtering']

        regions = measure.regionprops(labels, intensity_image=original_image)

        measurements = []
        final_labels_img = np.zeros_like(labels)
        nucleus_id_counter = 1

        img_min = original_image.min()
        img_max = original_image.max()
        if img_max > img_min:
             intensity_min_abs = img_min + (img_max - img_min) * filt_cfg['intensity_min_ratio']
        else:
             intensity_min_abs = img_min

        self.logger.info(f"Intensity Filter: Min mean intensity > {intensity_min_abs:.0f} (using ratio {filt_cfg['intensity_min_ratio']:.3f})")

        filtered_out_reasons = {'size': 0, 'circularity': 0, 'solidity': 0, 'eccentricity': 0, 'intensity': 0, 'perimeter': 0}

        for region in regions:
            # Filters
            if not (filt_cfg['size_min'] <= region.area <= filt_cfg['size_max']):
                filtered_out_reasons['size'] += 1; continue
            if region.perimeter == 0:
                 filtered_out_reasons['perimeter'] += 1; continue
            circularity = min(4 * np.pi * region.area / (region.perimeter ** 2), 1.0)
            if not (filt_cfg['circularity_min'] <= circularity <= filt_cfg['circularity_max']):
                 filtered_out_reasons['circularity'] += 1; continue
            if region.solidity < filt_cfg['solidity_min']:
                filtered_out_reasons['solidity'] += 1; continue
            if region.eccentricity > filt_cfg['eccentricity_max']:
                filtered_out_reasons['eccentricity'] += 1; continue
            if not hasattr(region, 'mean_intensity') or region.mean_intensity < intensity_min_abs:
                filtered_out_reasons['intensity'] += 1; continue

            # --- All filters passed. Measure properties. ---
            try:
                if not hasattr(region, 'intensity_image') or not hasattr(region, 'image') or region.intensity_image.shape != region.image.shape:
                     self.logger.warning(f"Region {region.label} missing attributes or shape mismatch. Using mean intensity.")
                     intensities = np.array([region.mean_intensity])
                else:
                     intensities = region.intensity_image[region.image]

                if intensities.size == 0:
                     self.logger.warning(f"Region {region.label} resulted in empty intensity array. Skipping.")
                     continue

                measurements.append({
                    'nucleus_id': nucleus_id_counter,
                    'area': region.area, 'perimeter': region.perimeter,
                    'circularity': circularity, 'solidity': region.solidity,
                    'eccentricity': region.eccentricity,
                    'major_axis': region.major_axis_length, 'minor_axis': region.minor_axis_length,
                    'centroid_x': region.centroid[1], 'centroid_y': region.centroid[0],
                    'mean_intensity': region.mean_intensity,
                    'integrated_density': np.sum(intensities),
                    'std_intensity': np.std(intensities),
                    'min_intensity': region.min_intensity if hasattr(region,'min_intensity') else np.min(intensities),
                    'max_intensity': region.max_intensity if hasattr(region,'max_intensity') else np.max(intensities),
                    'median_intensity': np.median(intensities)
                })

                final_labels_img[labels == region.label] = nucleus_id_counter
                nucleus_id_counter += 1
            except Exception as measure_exc:
                 self.logger.error(f"Error measuring properties for region {region.label}: {measure_exc}")
                 continue

        results_df = pd.DataFrame(measurements)
        self.logger.info(f"Found {len(regions)} total regions initially.")
        self.logger.info(f"Filtering breakdown: Size={filtered_out_reasons['size']}, Circularity={filtered_out_reasons['circularity']}, Solidity={filtered_out_reasons['solidity']}, Eccentricity={filtered_out_reasons['eccentricity']}, Intensity={filtered_out_reasons['intensity']}, Perimeter={filtered_out_reasons['perimeter']}")
        self.logger.info(f"Kept {len(results_df)} nuclei after filtering.")
        return results_df, final_labels_img


    def create_visualizations(self, original: np.ndarray, binary: np.ndarray,
                              distance: np.ndarray, final_labels: np.ndarray,
                              results_df: pd.DataFrame,
                              plots_output_path: Path,
                              file_stem: str):
        """Generates and saves the 4-panel diagnostic plot to a specified directory."""
        # create_visualizations remains the same
        if not self.config['visualization']['generate_all_plots']:
            return

        self.logger.info(f"Generating diagnostic visualizations to: {plots_output_path}")
        plots_output_path.mkdir(parents=True, exist_ok=True)

        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharex=True, sharey=True)

            # 1. Original
            vmax = np.percentile(original, 99.9)
            axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=vmax)
            axes[0, 0].set_title(f"1. Original 16-bit (Display: 0-{vmax})", fontsize=16)
            axes[0, 0].axis('off')

            # 2. Binary
            axes[0, 1].imshow(binary, cmap='gray')
            axes[0, 1].set_title("2. Binary Mask", fontsize=16)
            axes[0, 1].axis('off')

            # 3. Distance
            axes[1, 0].imshow(distance, cmap='magma')
            axes[1, 0].set_title("3. Distance Transform Map", fontsize=16)
            axes[1, 0].axis('off')

            # 4. Overlay
            overlay = label2rgb(
                final_labels, image=util.img_as_ubyte(original),
                bg_label=0, image_alpha=1.0, alpha=0.3, kind='overlay'
            )
            axes[1, 1].imshow(overlay)

            if not results_df.empty:
                for _, row in results_df.iterrows():
                    axes[1, 1].text(
                        row['centroid_x'], row['centroid_y'], str(row['nucleus_id']),
                        color='yellow', fontsize=8, ha='center', va='center'
                    )

            axes[1, 1].set_title(f"4. Final Count: {len(results_df)} Nuclei", fontsize=16)
            axes[1, 1].axis('off')

            plt.tight_layout()
            fig_path = plots_output_path / f"{file_stem}_diagnostic_plot.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"Diagnostic plot saved.")

        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")


    def count_dapi_nuclei_max_accuracy(self,
                                       image_path_str: str,
                                       output_dir: str, # Main output for measurements/config
                                       plots_output_path: str, # Separate dir for plots
                                       progress_callback=None) -> dict:
        """Main pipeline function."""
        # Main pipeline structure remains the same
        self.logger.info("Starting maximum accuracy DAPI nuclei analysis")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plots_path = Path(plots_output_path)

        image_path = Path(image_path_str)
        file_name_with_ext = image_path.name
        file_stem = image_path.stem

        self.logger.info(f"Processing: {file_name_with_ext}")

        try:
            # Step 1: Load
            original = self.load_image_16bit(image_path_str)
            if progress_callback: progress_callback("Loading image...", 20)

            # Step 2: Denoise
            denoised = self.advanced_denoise(original)
            if progress_callback: progress_callback("Denoising image...", 30)

            # Step 3: Threshold
            binary = self.multi_otsu_threshold(denoised)
            if progress_callback: progress_callback("Applying threshold...", 40)

            # Step 4: Morph Ops
            self.logger.info("Applying morphological operations...")
            size_min = self.config['filtering']['size_min']
            binary = morphology.remove_small_objects(binary, min_size=max(10, size_min // 20)) # Gentler cleaning
            binary = morphology.remove_small_holes(binary, area_threshold=max(25, size_min // 8)) # Gentler cleaning
            binary = morphology.binary_closing(binary, disk(2))
            if progress_callback: progress_callback("Morphological operations...", 50)

            # Step 5: Watershed
            labels, distance = self.enhanced_watershed(binary)
            if progress_callback: progress_callback("Watershed segmentation...", 60)

            # Step 6: Clear Border
            labels = segmentation.clear_border(labels)
            if progress_callback: progress_callback("Clearing border nuclei...", 70)

            # Step 7: Filter & Measure
            results_df, final_labels = self.measure_and_filter_nuclei(labels, original)
            if progress_callback: progress_callback("Filtering & measuring...", 80)

            # Generate Summary
            if results_df.empty:
                self.logger.warning("No nuclei passed all filters.")
                summary = {
                    'Slice': file_name_with_ext, 'Count': 0, 'Total Area': 0,
                    'Average Size': 0, '%Area': 0, 'Mean': 0
                }
            else:
                summary = {
                    'Slice': file_name_with_ext,
                    'Count': len(results_df),
                    'Total Area': results_df['area'].sum(),
                    'Average Size': results_df['area'].mean(),
                    '%Area': (results_df['area'].sum() / (original.size)) * 100,
                    'Mean': results_df['mean_intensity'].mean(),
                    '_std_area': results_df['area'].std(ddof=0),
                    '_std_intensity': results_df['mean_intensity'].std(ddof=0)
                }

            self.logger.info(f"=== MAXIMUM ACCURACY RESULTS ===")
            self.logger.info(f"Total Nuclei Count: {summary['Count']}")
            self.logger.info(f"Mean Area: {summary['Average Size']:.2f} pixel²")
            self.logger.info(f"Mean Intensity (16-bit): {summary['Mean']:.2f}")

            # --- Save Outputs (No individual summary CSV) ---
            csv_ind_path = output_path / f"{file_stem}_measurements_16bit.csv"
            results_df.to_csv(csv_ind_path, index=False, float_format='%.2f')
            self.logger.info(f"Full measurements saved to {csv_ind_path}")
            if progress_callback: progress_callback("Saving measurements...", 90)

            # Save config
            with open(output_path / f"{file_stem}_analysis_config.json", 'w') as f:
                config_to_save = {k: v for k, v in self.config.items() if isinstance(v, (dict, list, str, int, float, bool))}
                json.dump(config_to_save, f, indent=2)

            # Create Visualizations
            self.create_visualizations(
                original, binary, distance, final_labels,
                results_df,
                plots_path, # Use the dedicated path
                file_stem
            )
            if progress_callback: progress_callback("Creating visualizations...", 95)

            self.print_detailed_results(summary)

            self.logger.info(f"Analysis complete for {file_name_with_ext}. Measurements saved to: {output_path}")
            if progress_callback: progress_callback("Complete!", 100)

            return {
                'success': True,
                'summary': summary,
                'measurements': results_df.to_dict('records'),
                'output_dir': str(output_path)
            }

        except Exception as e:
            error_msg = f"Analysis failed for {file_name_with_ext}: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            if progress_callback: progress_callback(f"Error: {str(e)}", 0)

            return {
                'success': False,
                'error': error_msg,
                'summary': {'Slice': file_name_with_ext, 'Count': -1},
                'measurements': []
            }

    def print_detailed_results(self, summary: dict):
        """Print comprehensive results to console using corrected keys."""
        # print_detailed_results remains the same
        print("\n" + "="*70)
        print("MAXIMUM ACCURACY DAPI NUCLEI COUNTING RESULTS")
        print("="*70)
        print(f"Slice (Image Name): {summary['Slice']}")
        print(f"Total Nuclei Count: {summary['Count']}")
        print(f"Total Area: {summary.get('Total Area', 0):.0f} pixels²")
        print(f"Percentage Area: {summary.get('%Area', 0):.2f}%")
        print(f"Mean Nucleus Area: {summary.get('Average Size', 0):.2f} ± {summary.get('_std_area', 0):.2f} pixels²")
        print(f"Mean Intensity (16-bit): {summary.get('Mean', 0):.0f} ± {summary.get('_std_intensity', 0):.0f}")
        print("="*70)


# --- Progress Bar Window ---
class ProgressWindow:
    """A simple Tkinter window to show batch processing progress."""
    # Class definition remains the same
    def __init__(self, total_files):
        self.total_files = total_files
        self.current_file = 0
        self.root = tk.Toplevel()
        self.root.title("Processing Images...")
        self.root.geometry("450x120")
        self.root.resizable(False, False)
        self.root.transient()
        self.root.grab_set()
        self.progress_label = tk.Label(self.root, text="Starting batch process...", wraplength=400)
        self.progress_label.pack(pady=10, padx=10)
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5, padx=10)
        self.progress_bar["maximum"] = total_files
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        ws, hs = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x, y = (ws/2) - (w/2), (hs/2) - (h/2)
        self.root.geometry('+%d+%d' % (x, y))
        self.root.lift()
        self.root.attributes('-topmost',True)
        self.root.update()

    def update(self, filename):
        self.current_file += 1
        self.progress_bar["value"] = self.current_file
        percent = (self.current_file / self.total_files) * 100
        short_filename = Path(filename).name
        self.progress_label.config(text=f"Processing {self.current_file}/{self.total_files}: {short_filename} ({percent:.0f}%)")
        self.root.update_idletasks()

    def close(self):
        self.root.grab_release()
        self.root.destroy()


# ---
# === BATCH PROCESSING MODIFIED ===
# - Calculates proliferation percentage after processing all files.
# - Saves proliferation summary to a new CSV.
# ---
def batch_process(counter: MaximumAccuracyDAPICounter, input_folder: str, output_folder: str, default_params_filtering: dict):
    """
    Runs counter on sorted *_DAPI and *_CY5 images, saves plots centrally,
    creates ONE final batch summary CSV, and ONE proliferation summary CSV.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    plots_dir = output_path / "_diagnostic_plots"
    plots_dir.mkdir(exist_ok=True)

    counter.logger.info(f"--- Starting Batch Process ---")
    counter.logger.info(f"Input Folder: {input_path.resolve()}")
    counter.logger.info(f"Output Folder (Summary/Configs): {output_path.resolve()}")
    counter.logger.info(f"Diagnostic plots will be saved to: {plots_dir.resolve()}")

    # Find and sort files
    all_files = list(input_path.glob('*_DAPI.*')) + list(input_path.glob('*_CY5.*'))
    valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    image_files = [f for f in all_files if f.suffix.lower() in valid_extensions and f.is_file()]

    if not image_files:
        counter.logger.error(f"No valid _DAPI or _CY5 images found in {input_path.resolve()}")
        messagebox.showerror("Error", f"No valid _DAPI or _CY5 images found in the selected folder.")
        return

    def sort_key(filepath: Path):
        match = re.match(r"(\d+)_([A-Z0-9]+)", filepath.stem, re.IGNORECASE)
        if match:
            num = int(match.group(1)); channel = match.group(2).upper()
            channel_priority = 0 if channel == 'DAPI' else (1 if channel == 'CY5' else 2)
            return (num, channel_priority)
        return (float('inf'), 0)

    image_files.sort(key=sort_key)
    counter.logger.info(f"Found and sorted {len(image_files)} images for processing.")

    all_summaries = [] # Store individual image summary dicts
    successful_count = 0

    progress_win = ProgressWindow(len(image_files))

    try:
        for i, image_file in enumerate(image_files):
            counter.logger.info(f"\n--- Processing file {i+1}/{len(image_files)} ---")
            progress_win.update(image_file.name)

            measurement_output_dir = output_path / f"data_{image_file.stem}"

            # --- ADJUSTMENT LOGIC ---
            current_params_filtering = default_params_filtering.copy() # Start fresh each loop
            if '_CY5' in image_file.name.upper():
                # ===> ADJUSTMENT FOR CY5 <===
                cy5_intensity_ratio = 0.02 # Lowered significantly (adjust if needed)
                counter.logger.info(f"Applying adjusted intensity ratio ({cy5_intensity_ratio}) for CY5 image: {image_file.name}")
                current_params_filtering['intensity_min_ratio'] = cy5_intensity_ratio
            else:
                 # Ensure DAPI uses the default
                 current_params_filtering['intensity_min_ratio'] = default_params_filtering.get('intensity_min_ratio', 0.1)
            # -------------------------

            # Temporarily update config
            original_config_filtering = counter.config['filtering'].copy()
            counter.config['filtering'].update(current_params_filtering)

            result = counter.count_dapi_nuclei_max_accuracy(
                image_path_str=str(image_file),
                output_dir=str(measurement_output_dir),
                plots_output_path=str(plots_dir)
            )

            # Restore config
            counter.config['filtering'] = original_config_filtering

            if result['success']:
                successful_count += 1
                # Store the full summary dictionary including Slice and Count
                all_summaries.append(result['summary'])
                counter.logger.info(f"✓ Successfully processed {image_file.name}")
            else:
                counter.logger.error(f"✗ Failed to process {image_file.name}: {result.get('error', 'Unknown error')}")
    finally:
        progress_win.close()

    # --- Process Summaries for Batch CSV and Proliferation ---
    if all_summaries:
        batch_df = pd.DataFrame(all_summaries)
        
        # --- Save Standard Batch Summary ---
        batch_path = output_path / '_BATCH_SUMMARY_ALL.csv'
        user_columns = ['Slice', 'Count', 'Total Area', 'Average Size', '%Area', 'Mean']
        batch_df_to_save = batch_df.reindex(columns=[col for col in user_columns if col in batch_df.columns])
        batch_df_to_save.to_csv(batch_path, index=False, float_format='%.3f')
        counter.logger.info(f"\n--- Batch Summary Saved ---")
        counter.logger.info(f"Master summary saved to: {batch_path.resolve()}")

        # --- Calculate Proliferation ---
        proliferation_data = []
        # Group by the numerical prefix extracted from the 'Slice' column
        batch_df['Image_Set'] = batch_df['Slice'].str.extract(r'(\d+)_', expand=False)
        
        # Ensure Image_Set is treated numerically for correct grouping if needed later
        batch_df['Image_Set'] = pd.to_numeric(batch_df['Image_Set'], errors='coerce')
        batch_df.dropna(subset=['Image_Set'], inplace=True) # Drop rows where number couldn't be extracted
        batch_df['Image_Set'] = batch_df['Image_Set'].astype(int)

        grouped = batch_df.groupby('Image_Set')

        for name, group in grouped:
            dapi_row = group[group['Slice'].str.contains('_DAPI', case=False)]
            cy5_row = group[group['Slice'].str.contains('_CY5', case=False)]

            # Check if both DAPI and CY5 rows exist for this set
            if not dapi_row.empty and not cy5_row.empty:
                dapi_count = dapi_row.iloc[0]['Count']
                cy5_count = cy5_row.iloc[0]['Count']

                # Avoid division by zero
                if dapi_count > 0:
                    percentage = (cy5_count / dapi_count) * 100
                else:
                    percentage = 0 if cy5_count == 0 else np.inf # Or set to NaN if preferred

                proliferation_data.append({
                    'Image_Set': name,
                    'DAPI_Count': dapi_count,
                    'CY5_Count': cy5_count,
                    'Proliferation_Percentage': percentage
                })
            else:
                counter.logger.warning(f"Image set {name} is missing DAPI or CY5 count. Skipping proliferation calculation.")

        # --- Save Proliferation Summary ---
        if proliferation_data:
            proliferation_df = pd.DataFrame(proliferation_data)
            prolif_path = output_path / '_BATCH_PROLIFERATION_SUMMARY.csv'
            proliferation_df.to_csv(prolif_path, index=False, float_format='%.2f')
            counter.logger.info(f"Proliferation summary saved to: {prolif_path.resolve()}")
        else:
            counter.logger.warning("No complete DAPI/CY5 pairs found to calculate proliferation.")


        counter.logger.info(f"\n--- Batch Process Complete ---")
        counter.logger.info(f"Individual measurements & config files saved in 'data_*' subfolders within: {output_path.resolve()}")
        counter.logger.info(f"Diagnostic plots saved in '_diagnostic_plots' subfolder within: {output_path.resolve()}")

        messagebox.showinfo(
            "Batch Complete",
            f"Successfully processed {successful_count}/{len(image_files)} images\n"
            f"Batch summary & proliferation results saved in: {output_path.resolve()}"
        )
    else:
        counter.logger.warning("\n--- Batch Process Finished: No images were successfully processed. ---")
        messagebox.showwarning(
            "Batch Complete",
            f"No images were successfully processed.\n"
            f"Check the logs for error details."
        )


# --- Main Execution (Interactive) ---
def main():
    """Main execution function with GUI file selection"""

    # Default parameters dictionary (used for DAPI / Single File)
    default_analysis_params = {
        'sigma_spatial': 3,
        'sigma_color_ratio': 0.1,
        'otsu_classes': 2,
        'min_distance': 10,
        'size_min': 200,
        'size_max': 25000,
        'circularity_min': 0.40,
        'circularity_max': 1.00,
        'solidity_min': 0.85,
        'eccentricity_max': 0.9,
        'intensity_min_ratio': 0.1 # Default ratio for DAPI/Single file
    }

    # Create the config dictionary structure for the class
    config = {
        'processing': { k: default_analysis_params[k] for k in ['sigma_spatial', 'sigma_color_ratio', 'otsu_classes', 'min_distance']},
        'filtering': { k: default_analysis_params[k] for k in ['size_min', 'size_max', 'circularity_min', 'circularity_max', 'solidity_min', 'eccentricity_max', 'intensity_min_ratio']},
        'visualization': {'generate_all_plots': True}
    }

    # Initialize counter
    counter = MaximumAccuracyDAPICounter(config)

    root = tk.Tk()
    root.withdraw()

    print("=== Maximum Accuracy DAPI Nuclei Counter ===")
    print("Select processing mode:")
    print("1. Single image")
    print("2. Batch folder processing (processes *_DAPI.* and *_CY5.* files)")

    try:
        mode = input("Enter choice (1 or 2): ").strip()

        # Define base output directory
        if os.name == 'nt':
            output_base_parent = Path(os.environ.get('USERPROFILE', '.')) / "Documents"
        else:
            output_base_parent = Path(os.environ.get('HOME', '.'))

        if mode == '1':
            file_path = filedialog.askopenfilename(
                title="Select DAPI or CY5 image",
                filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
            )
            if not file_path: print("No file selected."); return

            print(f"\nProcessing: {file_path}")

            output_base = output_base_parent / "DAPI_Single_Analysis"
            output_base.mkdir(exist_ok=True)
            plots_dir = output_base / "_diagnostic_plots"
            measurement_dir = output_base / f"data_{Path(file_path).stem}"

            # --- ADJUSTMENT FOR SINGLE CY5 ---
            is_cy5 = '_CY5' in Path(file_path).name.upper()
            original_intensity_ratio = counter.config['filtering']['intensity_min_ratio'] # Store default
            if is_cy5:
                 cy5_intensity_ratio = 0.02 # Use lower value for single CY5 (Adjust if needed)
                 counter.logger.info(f"Single CY5 image detected. Adjusting intensity ratio to {cy5_intensity_ratio}")
                 counter.config['filtering']['intensity_min_ratio'] = cy5_intensity_ratio

            progress_win = ProgressWindow(1)
            progress_win.update(Path(file_path).name)

            def single_progress_callback(message, percentage):
                progress_win.root.title(f"Processing... {percentage:.0f}%")
                progress_win.progress_label.config(text=f"Status: {message}")
                progress_win.progress_bar['value'] = percentage
                progress_win.root.update()


            results = counter.count_dapi_nuclei_max_accuracy(
                image_path_str=file_path,
                output_dir=str(measurement_dir),
                plots_output_path=str(plots_dir),
                progress_callback=single_progress_callback
            )

            # Restore original config
            counter.config['filtering']['intensity_min_ratio'] = original_intensity_ratio

            progress_win.close()

            if results['success']:
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Counted {results['summary']['Count']} nuclei\n"
                    f"Results saved in subfolders within: {str(output_base.resolve())}"
                )
            else:
                messagebox.showerror(
                    "Analysis Failed",
                    f"Error processing {Path(file_path).name}:\n{results.get('error', 'Unknown error')}"
                )

        elif mode == '2':
            folder_path = filedialog.askdirectory(title="Select folder with DAPI and CY5 images")
            if not folder_path: print("No folder selected."); return

            output_base = output_base_parent / "DAPI_Batch_Analysis"
            output_base.mkdir(exist_ok=True)

            # Pass only the filtering part of the default parameters dictionary to batch_process
            batch_process(
                counter=counter,
                input_folder=folder_path,
                output_folder=str(output_base),
                default_params_filtering=config['filtering'] # Pass only filtering dict
            )

        else:
            print("Invalid choice. Please enter 1 or 2.")

    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
        logger_available = 'counter' in locals() and hasattr(counter, 'logger')
        if logger_available:
             counter.logger.error(error_msg)
        else:
             print(f"CRITICAL ERROR (Logger not init): {error_msg}")
        messagebox.showerror("Critical Error", f"An unexpected error occurred. Check logs or console output for details:\n{str(e)}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()