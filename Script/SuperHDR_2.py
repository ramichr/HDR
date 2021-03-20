import cv2
import numpy as np
import imageio

from Script.SuperHDR_aux import count_pixels, sort_into_chain, adjust_exposure, trinarize, trinarize_vis, align_image, \
    merging


# Determines reference image based on most appropriate pixels
# Sorts the images into an exposure series from darkest to brightest


def preprocess(sdr_series):
    ref = sdr_series[0]
    for sdr_image in sdr_series:
        count_pixels(sdr_image)
        if ref.pixels_appr < sdr_image.pixels_appr:
            ref = sdr_image
    print("ref exposure", ref.relative_exp)
    for sdr_image in sdr_series:
        sort_into_chain(ref, sdr_image)
    return ref


# Normalizes the luminance channel of an image in relation to the reference image
def normalize_luminance(sdr_series, ref):
    for sdr_image in sdr_series:
        if sdr_image != ref:
            adjust_exposure(sdr_image, ref)


# Aligns all images with the reference image as good as possible
def align_images_to_ref(sdr_series, ref):
    trinarize(ref)
    cv2.imshow("ref tri", trinarize_vis(ref.trinarized))
    for sdr_image in sdr_series:
        if sdr_image != ref:
            trinarize(sdr_image)
            sdr_image.displacement = align_image(sdr_image.trinarized, ref.trinarized)
            print(sdr_image.displacement)


# Executes all steps for HDR imaging and outputs result to test.exr
def sdr_series_to_hdr(sdr_series):
    ref = preprocess(sdr_series)
    normalize_luminance(sdr_series, ref)
    align_images_to_ref(sdr_series, ref)
    hdr = merging(ref)
    hdr_rgb = hdr[..., ::-1].copy()
    imageio.imwrite("test.exr", hdr_rgb, "exr")


def sdr_series_to_hdr_array(sdr_series):
    ref = preprocess(sdr_series)
    normalize_luminance(sdr_series, ref)
    align_images_to_ref(sdr_series, ref)
    hdr = merging(ref)
    return hdr[..., ::-1].copy()


# Imports the standard DR exposure series
# parliament for test set of a parliament and water
# parliament2 for a (digitally) displaced test set
# opencv_test_set for the image set used for OpenCVHDR
# Our own sets: own_monster, own_monster_mini, own_street, own_street, own_street_moved, own_street_moved_enhanced,
# own_statue, own_street_moved_mini

#sdr_series = import_series("own_street")

# Starts the HDR imaging process
# The process will output the relative exposure of the reference image for identification
# as well as the displacements of the other images. After that, it might take some time before
# the window shows the trinarized reference image. At this point, the algorithm will have generated a test.exr file.

#sdr_series_to_hdr(sdr_series)


#cv2.waitKey(0)
