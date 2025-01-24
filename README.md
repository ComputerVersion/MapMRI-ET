# MapMRI-ET

## Overview
This project contains a set of Python scripts related to MRI (Magnetic Resonance Imaging) data processing. The scripts are designed to handle various tasks such as data loading, brain extraction, fitting data to specific models, and saving processed images.

The example data is contained in the `MAP-MRI-data` folder and the results visualization is in the `results-visualization` folder.

## Scripts Introduction

### 1. MapMRI.py
This script contains several utility functions for handling MRI data, including data loading, gradient table generation, brain extraction, model fitting, and image saving.

#### Functions
- **`get_data(niiPath)`**:
    - **Functionality**: Loads NIfTI data from the given file path using `load_nifti` function from `dipy.io.image`.
    - **Parameters**:
        - `niiPath`: The file path of the NIfTI data file.
    - **Returns**: A tuple containing the loaded MRI data and its affine transformation matrix.

- **`get_bvals_bvecs(bvalsPath, bvecsPath)`**:
    - **Functionality**: Reads b-values and b-vectors from the given file paths using `read_bvals_bvecs` function from `dipy.io.gradients`.
    - **Parameters**:
        - `bvalsPath`: The file path of the b-values file.
        - `bvecsPath`: The file path of the b-vectors file.
    - **Returns**: A tuple containing the read b-values and b-vectors.

- **`get_gtab(bvals, bvecs, big_delta, small_delta)`**:
    - **Functionality**: Creates a gradient table object. First, it creates a basic gradient table using the given b-values and b-vectors. Then, it creates a new gradient table with additional `big_delta` and `small_delta` parameters using the b-values and b-vectors from the first gradient table.
    - **Parameters**:
        - `bvals`: The b-values.
        - `bvecs`: The b-vectors.
        - `big_delta`: The big delta parameter for the gradient table.
        - `small_delta`: The small delta parameter for the gradient table.
    - **Returns**: The final gradient table object.

- **`brain_extraction(data)`**:
    - **Functionality**: Performs brain extraction on the given MRI data. It first removes single-dimensional entries from the shape of the data using `np.squeeze`. Then, it calculates a parameter based on the shape of the data. Finally, it applies the `median_otsu` function from `dipy.segment.mask` to extract the brain region.
    - **Parameters**:
        - `data`: The MRI data.
    - **Returns**: The extracted brain data.

- **`Laplacian(gtab, radial_order)`**:
    - **Functionality**: Creates a `MapmriModel` object with Laplacian regularization. The regularization weight is set to 0.2.
    - **Parameters**:
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the `MapmriModel`.
    - **Returns**: The created `MapmriModel` object.

- **`Positivity(gtab, radial_order)`**:
    - **Functionality**: Creates a `MapmriModel` object with positivity constraint. Laplacian regularization is disabled, and the `cvxpy_solver` is set to 'MOSEK'.
    - **Parameters**:
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the `MapmriModel`.
    - **Returns**: The created `MapmriModel` object.

- **`BothMethod(gtab, radial_order)`**:
    - **Functionality**: Creates a `MapmriModel` object with both Laplacian regularization and positivity constraint. The Laplacian weighting is set to 0.1, and the `cvxpy_solver` is set to 'MOSEK'.
    - **Parameters**:
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the `MapmriModel`.
    - **Returns**: The created `MapmriModel` object.

- **`BothMethod_NG(gtab, radial_order)`**:
    - **Functionality**: Creates a `MapmriModel` object with both Laplacian regularization and positivity constraint, and a b-value threshold of 2000. The Laplacian weighting is set to 0.1, and the `cvxpy_solver` is set to 'MOSEK'.
    - **Parameters**:
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the `MapmriModel`.
    - **Returns**: The created `MapmriModel` object.

- **`Laplacian_NG(gtab, radial_order)`**:
    - **Functionality**: Creates a `MapmriModel` object with Laplacian regularization, a Laplacian weighting of 0.05, and positivity constraint disabled. A b-value threshold of 2000 is also set.
    - **Parameters**:
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the `MapmriModel`.
    - **Returns**: The created `MapmriModel` object.

- **`Positivity_NG(gtab, radial_order)`**:
    - **Functionality**: Creates a `MapmriModel` object with positivity constraint, Laplacian regularization disabled, a Laplacian weighting of 0.05, and a b-value threshold of 2000. The `cvxpy_solver` is set to 'MOSEK'.
    - **Parameters**:
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the `MapmriModel`.
    - **Returns**: The created `MapmriModel` object.

- **`BothMethod_ODF(gtab, radial_order)`**:
    - **Functionality**: Creates a `MapmriModel` object with both Laplacian regularization and positivity constraint, anisotropic scaling disabled, and a fixed radial order of 8. The Laplacian weighting is set to 0.1.
    - **Parameters**:
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the `MapmriModel` (although it is overwritten to 8 inside the function).
    - **Returns**: The created `MapmriModel` object.

- **`getFitData(data, method, algorithm, gtab, radial_order)`**:
    - **Functionality**: Fits the MRI data using the appropriate model based on the given `method` and `algorithm`. It calls different model creation functions (`Laplacian`, `Positivity`, `BothMethod`, etc.) depending on the input.
    - **Parameters**:
        - `data`: The MRI data.
        - `method`: The fitting method, which can be 'laplacian', 'positivity', or 'both'.
        - `algorithm`: The algorithm, which can be 'aniso' or 'ng'.
        - `gtab`: The gradient table object.
        - `radial_order`: The radial order for the model.
    - **Returns**: The fitted data.

- **`saveImage(image_data, image_name, method, algorithm, niiPath, affine = None, use_int16 = False)`**:
    - **Functionality**: Saves the given image data as a NIfTI file in the same folder as the original NIfTI file. If `use_int16` is `True`, it converts the image data to `np.int16` type. It creates a new folder based on the `method` if it doesn't exist and saves the image with a specific name.
    - **Parameters**:
        - `image_data`: The image data to be saved.
        - `image_name`: The name of the image.
        - `method`: The method used for the image.
        - `algorithm`: The algorithm used for the image.
        - `niiPath`: The file path of the original NIfTI file.
        - `affine`: The affine transformation matrix (optional).
        - `use_int16`: Whether to convert the image data to `np.int16` type (default is `False`).
    - **Returns**: None.

- **`saveImageBoth(image_data, image_name, method, algorithm, niiPath, Weight, affine = None, use_int16 = False)`**:
    - **Functionality**: Similar to `saveImage`, but it appends the `Weight` value to the `method` name when creating the folder and the file name.
    - **Parameters**:
        - `image_data`: The image data to be saved.
        - `image_name`: The name of the image.
        - `method`: The method used for the image.
        - `algorithm`: The algorithm used for the image.
        - `niiPath`: The file path of the original NIfTI file.
        - `Weight`: The weight value to be included in the folder and file names.
        - `affine`: The affine transformation matrix (optional).
        - `use_int16`: Whether to convert the image data to `np.int16` type (default is `False`).
    - **Returns**: None.


### 2. MapMRI_batch.py
This script is used for batch processing of MRI data.

#### Main Function
- **`MapMRI_Batch(path, method, algorithm, Weight)`**:
    - **Functionality**: Iterates through patient folders in the given path, processes each patient's MRI data, fits the data to a specific model, and saves the processed images.
    - **Parameters**:
        - `path`: The main path containing patient folders.
        - `method`: The method to be used, which can be 'laplacian', 'positivity', or 'both'.
        - `algorithm`: The algorithm to be used, which can be 'aniso' or 'ng'.
        - `Weight`: The weight value, only used when `method` is 'both'.


### 3. MapMRI_main.py
This script contains the main class `MapMRI_Fit` for fitting MRI data and saving processed images.

#### Class: MapMRI_Fit
- **`__init__(self)`**: Initializes the class.
- **`fitData(self, data, method, algorithm, gtab, radial_order)`**:
    - **Functionality**: Fits the given data to a specific model.
    - **Parameters**:
        - `data`: The MRI data.
        - `method`: The method to be used, such as 'laplacian', 'positivity', or 'both'.
        - `algorithm`: The algorithm to be used, such as 'aniso' or 'ng'.
        - `gtab`: The gradient table.
        - `radial_order`: The expansion order of the basis.
    - **Returns**: The fitted image data.
- **`saveRtop(self, image_data, method, algorithm, niiPath, affine)`**:
    - **Functionality**: Saves the processed rtop image data.
    - **Parameters**:
        - `image_data`: The image data to be saved.
        - `method`: The method used for processing.
        - `algorithm`: The algorithm used for processing.
        - `niiPath`: The file path for saving.
        - `affine`: The affine transformation matrix.
- **`saveRtap(self, image_data, method, algorithm, niiPath, affine)`**: Similar to `saveRtop`, but for rtap image data.
- **`saveRtpp(self, image_data, method, algorithm, niiPath, affine)`**: Similar to `saveRtop`, but for rtpp image data.
- **`saveMsd(self, image_data, method, algorithm, niiPath, affine)`**: Similar to `saveRtop`, but for msd image data.
- **`saveQiv(self, image_data, method, algorithm, niiPath, affine)`**: Similar to `saveRtop`, but for qiv image data.
- **`saveNg(self, image_data, method, algorithm, niiPath, affine)`**: Similar to `saveRtop`, but for ng image data.
- **`saveNgParallel(self, image_data, method, algorithm, niiPath, affine)`**: Similar to `saveRtop`, but for ng parallel image data.
- **`saveNgPerpendicular(self, image_data, method, algorithm, niiPath, affine)`**: Similar to `saveRtop`, but for ng perpendicular image data.
- **`saveAnisoAll(self, image_data, method, algorithm, niiPath, affine)`**:
    - **Functionality**: Saves all "aniso" images.
    - **Parameters**:
        - `image_data`: The image data to be saved.
        - `method`: The method used for processing.
        - `algorithm`: The algorithm used for processing.
        - `niiPath`: The file path for saving.
        - `affine`: The affine transformation matrix.
- **`saveNgAll(self, image_data, method, algorithm, niiPath, affine)`**:
    - **Functionality**: Saves all "ng" images.
    - **Parameters**: Similar to `saveAnisoAll`.
- **`saveAnisoAllBoth(self, image_data, method, algorithm, niiPath, Weight, affine)`**:
    - **Functionality**: Saves all "aniso" images with a specific weight.
    - **Parameters**: Similar to `saveAnisoAll`, with an additional `Weight` parameter.
- **`saveNgAllBoth(self, image_data, method, algorithm, niiPath, Weight, affine)`**:
    - **Functionality**: Saves all "ng" images with a specific weight.
    - **Parameters**: Similar to `saveNgAll`, with an additional `Weight` parameter.

## Dependencies
- Python 3.x
-Libraries: `numpy`, `amico`, `dipy`, `nibabel`, `matplotlib`, `mpl_toolkits`

## Usage
You can run the scripts in a Python environment. For example, to run the batch processing script, you can call the `MapMRI_Batch` function in `MapMRI_batch.py` with appropriate parameters.
```bash
python mrimap/MapMRI_batch.py
```