from MapMRI_main import MapMRI_Fit
import MapMRI
import os

# gradient parameters
big_delta =0.038
small_delta =0.0154

# expansion order of the basis, i.e. how many basis functions to approximate the signal
radial_order = 4

# main path
path = r"D:\MAPMRI\map_nii"


# choose method from 'laplacian', 'positivity' or 'both'
method = 'laplacian'
# choose algorithm from 'aniso' or 'ng'
algorithm = 'ng'

Weight = 0.1   # only for both ()
 
def MapMRI_Batch(path, method, algorithm,Weight):

    for patient_folder in os.listdir(path):    # go through every patient's folder
        patient_path = os.path.join(path, patient_folder)
        for file in os.listdir(patient_path):    # get file paths for each patient
            file_path = os.path.join(patient_path, file)
            if file.endswith(".nii"):
                niiPath = file_path
                print(niiPath)
            elif file.endswith(".bval"):
                bvalsPath = file_path
                print(bvalsPath)
            elif file.endswith(".bvec"):
                bvecsPath = file_path
                print(bvecsPath)

        data, affine = MapMRI.get_data(niiPath)    # get data and affine
        data = MapMRI.brain_extraction(data)    # add mask to data
        bvals, bvecs = MapMRI.get_bvals_bvecs(bvalsPath, bvecsPath)    # get b-values and b-vectors
        gtab = MapMRI.get_gtab(bvals, bvecs, big_delta, small_delta)    # get gradient information
        # print('data.shape (%d, %d, %d, %d)' % data.shape)


        # fit data
        image_data = MapMRI_Fit().fitData(data, method, algorithm, gtab, radial_order)

        if method == 'both':

            # save images
            if algorithm == 'aniso':
                MapMRI_Fit().saveAnisoAllBoth(image_data, method, algorithm, niiPath, Weight,affine)
            else:
                MapMRI_Fit().saveNgAllBoth(image_data, method, algorithm, niiPath, Weight, affine)
        else:
            # save images
            if algorithm == 'aniso':
                MapMRI_Fit().saveAnisoAll(image_data, method, algorithm, niiPath, affine)
            else:
                MapMRI_Fit().saveNgAll(image_data, method, algorithm, niiPath, affine)

MapMRI_Batch(path, method, algorithm, Weight)