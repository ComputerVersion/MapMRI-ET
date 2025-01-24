# -*- coding: utf-8 -*-
# https://docs.dipy.org/stable/reference/dipy.reconst.html#module-dipy.reconst.mapmri
from dipy.reconst import mapmri
from dipy.viz import window, actor
from dipy.data import get_fnames, get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dipy.segment.mask import median_otsu
import os

def get_data(niiPath):

    data, affine = load_nifti(niiPath)
    return data, affine


def get_bvals_bvecs(bvalsPath, bvecsPath):

    bvals, bvecs = read_bvals_bvecs(bvalsPath, bvecsPath)
    return bvals,bvecs


def get_gtab(bvals,bvecs,big_delta,small_delta):
    gtab = gradient_table(bvals, bvecs)
    gtab = gradient_table(bvals=gtab.bvals, bvecs=gtab.bvecs,
                          big_delta=big_delta,
                          small_delta=small_delta)

    return gtab

def brain_extraction(data):
    data = np.squeeze(data)
    dataShape = data.shape
    param = dataShape[3] - 1
    brain, mask = median_otsu(data, vol_idx=[param], median_radius=4, numpass=4, autocrop=True)
    return brain

# data_small = data[40:65, 50:51]

# print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
The MAPMRI Model can now be instantiated. The ``radial_order`` determines the
expansion order of the basis, i.e., how many basis functions are used to
approximate the signal.

First, we must decide to use the anisotropic or isotropic MAPMRI basis. As was
shown in [Fick2016a]_, the isotropic basis is best used for tractography
purposes, as the anisotropic basis has a bias towards smaller crossing angles
in the ODF. For signal fitting and estimation of scalar quantities the
anisotropic basis is suggested. The choice can be made by setting
``anisotropic_scaling=True`` or ``anisotropic_scaling=False``.

First, we must select the method of regularization and/or constraining the
basis fitting.

- ``laplacian_regularization=True`` makes it use Laplacian regularization
  [Fick2016a]_. This method essentially reduces spurious oscillations in the
  reconstruction by minimizing the Laplacian of the fitted signal.
  Several options can be given to select the regularization weight:

    - ``regularization_weighting=GCV`` uses generalized cross-validation
      [Craven1978]_ to find an optimal weight.
    - ``regularization_weighting=0.2`` for example would omit the GCV and
      just set it to 0.2 (found to be reasonable in HCP data [Fick2016a]_).
    - ``regularization_weighting=np.array(weights)`` would make the GCV use
      a custom range to find an optimal weight.

- ``positivity_constraint=True`` makes it use the positivity constraint on the
  diffusion propagator [Ozarslan2013]_. This method constrains the final
  solution of the diffusion propagator to be positive at a set of discrete
  points, since negative values should not exist.

    - The ``pos_grid`` and ``pos_radius`` affect the location and number of
      constraint points in the diffusion propagator.


"""
def Laplacian(gtab,radial_order):

    map_model_laplacian_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                   laplacian_regularization=True,
                                                   laplacian_weighting=.2)
    return map_model_laplacian_aniso


def Positivity(gtab,radial_order):

    map_model_positivity_aniso = mapmri.MapmriModel(gtab,
                                                    radial_order=radial_order,
                                                    laplacian_regularization=False,
                                                    positivity_constraint=True,
                                                    cvxpy_solver='MOSEK')
    return map_model_positivity_aniso


def BothMethod(gtab,radial_order):

    map_model_both_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                              laplacian_regularization=True,
                                              laplacian_weighting=.1,
                                              positivity_constraint=True,
                                              cvxpy_solver='MOSEK')
    return map_model_both_aniso


def BothMethod_NG(gtab,radial_order):

    map_model_both_ng = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                           laplacian_regularization=True,
                                           laplacian_weighting=.1,
                                           positivity_constraint=True,
                                           bval_threshold=2000,
                                           cvxpy_solver='MOSEK')
    return map_model_both_ng


def Laplacian_NG(gtab,radial_order):

    map_model_both_ng = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                           laplacian_regularization=True,
                                           laplacian_weighting=.05,
                                           positivity_constraint=False,
                                           bval_threshold=2000)
    return map_model_both_ng



def Positivity_NG(gtab,radial_order):

    map_model_both_ng = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                           laplacian_regularization=False,
                                           laplacian_weighting=.05,
                                           positivity_constraint=True,
                                           bval_threshold=2000,
                                           cvxpy_solver='MOSEK')
    return map_model_both_ng



def BothMethod_ODF(gtab,radial_order):

    radial_order = 8
    map_model_both_iso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                            laplacian_regularization=True,
                                            laplacian_weighting=.1,
                                            positivity_constraint=True,
                                            anisotropic_scaling=False)


    return map_model_both_iso

def getFitData(data, method, algorithm, gtab, radial_order):
    if method == 'laplacian':
        if algorithm == 'aniso':
           return Laplacian(gtab, radial_order).fit(data)
        elif algorithm == 'ng':
            return Laplacian_NG(gtab, radial_order).fit(data)
    elif method == 'positivity':
        if algorithm == 'aniso':
           return Positivity(gtab, radial_order).fit(data)
        elif algorithm == 'ng':
            return Positivity_NG(gtab, radial_order).fit(data)
    elif method == 'both':
        if algorithm == 'aniso':
           return BothMethod(gtab, radial_order).fit(data)
        elif algorithm == 'ng':
            return BothMethod_NG(gtab, radial_order).fit(data)         

def saveImage(image_data, image_name, method, algorithm, niiPath, affine = None, use_int16 = False):
    """
    Given fitted data, this function will save image data as ".nii" format under the same folder where the original ".nii" file comes from.
    """
    if use_int16 == True:   # For certain image data, we need to transfer it to int16 type to make the image show better.
        image_data = image_data.astype(np.int16)
    image = nib.Nifti1Image(image_data, affine)
    dataName = os.path.basename(niiPath) 
    fileName = os.path.splitext(dataName)[0]
    folderPath = os.path.join(os.path.abspath(os.path.join(niiPath,"..")), method)
    os.makedirs(folderPath, exist_ok = True)
    name = fileName + "_" + method + "_" + algorithm + "_" + image_name+ ".nii"
    path_save = os.path.join(folderPath, name)
    # path_save = os.path.abspath(os.path.join(niiPath,"..")) + "\\" + os.path.split(niiPath)[1] + "." + method + "_" + algorithm + "_" + image_name+ ".nii"
    nib.save(image, path_save)

def saveImageBoth(image_data, image_name, method, algorithm, niiPath, Weight, affine = None, use_int16 = False):
    """
    Given fitted data, this function will save image data as ".nii" format under the same folder where the original ".nii" file comes from.
    """
    if use_int16 == True:   # For certain image data, we need to transfer it to int16 type to make the image show better.
        image_data = image_data.astype(np.int16)
    image = nib.Nifti1Image(image_data, affine)
    dataName = os.path.basename(niiPath)
    fileName = os.path.splitext(dataName)[0]
    method = method + str(Weight)
    folderPath = os.path.join(os.path.abspath(os.path.join(niiPath,"..")), method)
    os.makedirs(folderPath, exist_ok = True)
    name = fileName + "_" + method + "_" + algorithm + str(Weight) + "_" + image_name+ ".nii"
    path_save = os.path.join(folderPath, name)
    # path_save = os.path.abspath(os.path.join(niiPath,"..")) + "\\" + os.path.split(niiPath)[1] + "." + method + "_" + algorithm + "_" + image_name+ ".nii"
    nib.save(image, path_save)



# def showRTOP(mapfit_laplacian_aniso,mapfit_positivity_aniso,mapfit_both_aniso):
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(1, 3, 1, title=r'RTOP - Laplacian')
#     ax1.set_axis_off()
#     rtop_laplacian = np.array(mapfit_laplacian_aniso.rtop()[:, 0, :].T,
#                               dtype=float)
#     ind = ax1.imshow(rtop_laplacian, interpolation='nearest',
#                      origin='lower', cmap=plt.cm.gray)

#     ax2 = fig.add_subplot(1, 3, 2, title=r'RTOP - Positivity')
#     ax2.set_axis_off()
#     rtop_positivity = np.array(mapfit_positivity_aniso.rtop()[:, 0, :].T,
#                                dtype=float)
#     ind = ax2.imshow(rtop_positivity, interpolation='nearest',
#                      origin='lower', cmap=plt.cm.gray)

#     ax3 = fig.add_subplot(1, 3, 3, title=r'RTOP - Both')
#     ax3.set_axis_off()
#     rtop_both = np.array(mapfit_both_aniso.rtop()[:, 0, :].T, dtype=float)
#     ind = ax3.imshow(rtop_both, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)
#     divider = make_axes_locatable(ax3)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(ind, cax=cax)

#     plt.savefig('MAPMRI_maps_regularization.png')


# def showlpNorm(mapfit_laplacian_aniso,mapfit_positivity_aniso,mapfit_both_aniso):
#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(1, 3, 1, title=r'Laplacian norm - Laplacian')
#     ax1.set_axis_off()
#     laplacian_norm_laplacian = np.array(mapfit_laplacian_aniso.norm_of_laplacian_signal()[:, 0, :].T,
#                                         dtype=float)
#     ind = ax1.imshow(laplacian_norm_laplacian, interpolation='nearest',
#                      origin='lower', cmap=plt.cm.gray, vmin=0, vmax=3)

#     ax2 = fig.add_subplot(1, 3, 2, title=r'Laplacian norm - Positivity')
#     ax2.set_axis_off()
#     laplacian_norm_positivity = np.array(mapfit_positivity_aniso.norm_of_laplacian_signal()[:, 0, :].T,
#                                          dtype=float)
#     ind = ax2.imshow(laplacian_norm_positivity, interpolation='nearest',
#                      origin='lower', cmap=plt.cm.gray, vmin=0, vmax=3)

#     ax3 = fig.add_subplot(1, 3, 3, title=r'Laplacian norm - Both')
#     ax3.set_axis_off()
#     laplacian_norm_both = np.array(mapfit_both_aniso.norm_of_laplacian_signal()[:, 0, :].T,
#                                    dtype=float)
#     ind = ax3.imshow(laplacian_norm_both, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray, vmin=0, vmax=3)
#     divider = make_axes_locatable(ax3)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(ind, cax=cax)
#     plt.savefig('MAPMRI_norm_laplacian.png')


# def showAll(mapfit_laplacian_aniso,mapfit_positivity_aniso,mapfit_both_aniso):
#     fig = plt.figure(figsize=(15, 6))
#     ax1 = fig.add_subplot(1, 5, 1, title=r'MSD')
#     ax1.set_axis_off()
#     msd = np.array(mapfit_both_aniso.msd()[:, 0, :].T, dtype=float)
#     ind = ax1.imshow(msd, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)

#     ax2 = fig.add_subplot(1, 5, 2, title=r'QIV')
#     ax2.set_axis_off()
#     qiv = np.array(mapfit_both_aniso.qiv()[:, 0, :].T, dtype=float)
#     ind = ax2.imshow(qiv, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)

#     ax3 = fig.add_subplot(1, 5, 3, title=r'RTOP')
#     ax3.set_axis_off()
#     rtop = np.array((mapfit_both_aniso.rtop()[:, 0, :]).T, dtype=float)
#     ind = ax3.imshow(rtop, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)

#     ax4 = fig.add_subplot(1, 5, 4, title=r'RTAP')
#     ax4.set_axis_off()
#     rtap = np.array((mapfit_both_aniso.rtap()[:, 0, :]).T, dtype=float)
#     ind = ax4.imshow(rtap, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)

#     ax5 = fig.add_subplot(1, 5, 5, title=r'RTPP')
#     ax5.set_axis_off()
#     rtpp = np.array(mapfit_both_aniso.rtpp()[:, 0, :].T, dtype=float)
#     ind = ax5.imshow(rtpp, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)
#     plt.savefig('MAPMRI_maps.png')


# def showNG(mapfit_both_ng):


#     fig = plt.figure(figsize=(10, 6))
#     ax1 = fig.add_subplot(1, 3, 1, title=r'NG')
#     ax1.set_axis_off()
#     ng = np.array(mapfit_both_ng.ng()[:, 0, :].T, dtype=float)
#     ind = ax1.imshow(ng, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)
#     divider = make_axes_locatable(ax1)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(ind, cax=cax)

#     ax2 = fig.add_subplot(1, 3, 2, title=r'NG perpendicular')
#     ax2.set_axis_off()
#     ng_perpendicular = np.array(mapfit_both_ng.ng_perpendicular()[:, 0, :].T,
#                                 dtype=float)
#     ind = ax2.imshow(ng_perpendicular, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)
#     divider = make_axes_locatable(ax2)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(ind, cax=cax)

#     ax3 = fig.add_subplot(1, 3, 3, title=r'NG parallel')
#     ax3.set_axis_off()
#     ng_parallel = np.array(mapfit_both_ng.ng_parallel()[:, 0, :].T, dtype=float)
#     ind = ax3.imshow(ng_parallel, interpolation='nearest', origin='lower',
#                      cmap=plt.cm.gray)
#     divider = make_axes_locatable(ax3)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(ind, cax=cax)
#     plt.savefig('MAPMRI_ng.png')


# def showODF(mapfit_both_iso):

#     sphere = get_sphere('repulsion724')

#     odf = mapfit_both_iso.odf(sphere, s=2)
#     print('odf.shape (%d, %d, %d, %d)' % odf.shape)

#     interactive = False

#     scene = window.Scene()
#     sfu = actor.odf_slicer(odf, sphere=sphere, colormap='plasma', scale=0.5)
#     sfu.display(y=0)
#     sfu.RotateX(-90)
#     scene.add(sfu)
#     window.record(scene, out_path='odfs.png', size=(600, 600))
#     if interactive:
#         window.show(scene)


# def getODF(mapfit_both_iso):

#     sphere = get_sphere('repulsion724')

#     odf = mapfit_both_iso.odf(sphere, s=2)

#     return odf

