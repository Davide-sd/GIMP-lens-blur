#!/usr/bin/python

# GIMP Lens Blur
# This plugin implements the Lens Blur, also known as Bokeh Blur.
# The base code has been implemented by Mike Pound: https://github.com/mikepound/convolve
# This plugin uses OpenCV instead of scipy.signal for performance reasons.

# Copyright (c) 2019 Davide Sandona'
# sandona [dot] davide [at] gmail [dot] com
# https://github.com/Davide-sd/GIMP-lens-blur.git

# Many thanks to Niklas Liebig for the alpha channel correction! :)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

from gimpfu import *
import numpy as np
import cv2
from functools import reduce

def channelData(layer):
    """ Returns a numpy array of the size [height, width, bpp] of the input layer.
    bpp stands for Bytes Per Pixel.
    """
    w, h = layer.width, layer.height
    region = layer.get_pixel_rgn(0, 0, w, h)
    pixChars = region[:, :]
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(h, w, bpp)

def createResultLayer(image, name, result):
    """ Create and add a new layer to the image.
    Input parameters:
        image   : the image where to add the new layer
        name    : the layer name
        result  : the pixels color informations for the new layer
    """
    rlBytes = np.uint8(result).tobytes()
    rl = gimp.Layer(image, name, result.shape[1], result.shape[0],
                  image.active_layer.type, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()

################################################################################
########### https://github.com/mikepound/convolve/complex_kernels.py ###########
################################################################################

# These scales bring the size of the below components to roughly the specified radius - I just hard coded these
kernel_scales = [1.4,1.2,1.2,1.2,1.2,1.2]

# Kernel parameters a, b, A, B
# These parameters are drawn from <http://yehar.com/blog/?p=1495>
kernel_params = [
                # 1-component
                [[0.862325, 1.624835, 0.767583, 1.862321]],

                # 2-components
                [[0.886528, 5.268909, 0.411259, -0.548794],
                [1.960518, 1.558213, 0.513282, 4.56111]],

                # 3-components
                [[2.17649, 5.043495, 1.621035, -2.105439],
                [1.019306, 9.027613, -0.28086, -0.162882],
                [2.81511, 1.597273, -0.366471, 10.300301]],

                # 4-components
                [[4.338459, 1.553635, -5.767909, 46.164397],
                [3.839993, 4.693183, 9.795391, -15.227561],
                [2.791880, 8.178137, -3.048324, 0.302959],
                [1.342190, 12.328289, 0.010001, 0.244650]],

                # 5-components
                [[4.892608, 1.685979, -22.356787, 85.91246],
                [4.71187, 4.998496, 35.918936, -28.875618],
                [4.052795, 8.244168, -13.212253, -1.578428],
                [2.929212, 11.900859, 0.507991, 1.816328],
                [1.512961, 16.116382, 0.138051, -0.01]],

                # 6-components
                [[5.143778, 2.079813, -82.326596, 111.231024],
                [5.612426, 6.153387, 113.878661, 58.004879],
                [5.982921, 9.802895, 39.479083, -162.028887],
                [6.505167, 11.059237, -71.286026, 95.027069],
                [3.869579, 14.81052, 1.405746, -3.704914],
                [2.201904, 19.032909, -0.152784, -0.107988]]]

# Obtain specific parameters and scale for a given component count
def get_parameters(component_count = 2):
    # make sure this condition is respected: 0 <= parameter_index < len(kernel_params)
    parameter_index = int(max(0, min(component_count - 1, len(kernel_params) - 1)))
    parameter_dictionaries = [dict(zip(['a','b','A','B'], b)) for b in kernel_params[parameter_index]]
    return (parameter_dictionaries, kernel_scales[parameter_index])

# Produces a complex kernel of a given radius and scale (adjusts radius to be more accurate)
# a and b are parameters of this complex kernel
def complex_kernel_1d(radius, scale, a, b):
    kernel_radius = radius
    kernel_size = int(kernel_radius * 2 + 1)
    ax = np.arange(-kernel_radius, kernel_radius + 1., dtype=np.float32)
    ax = ax * scale * (1 / kernel_radius)
    kernel_complex = np.zeros((kernel_size), dtype=np.complex64)
    kernel_complex.real = np.exp(-a * (ax**2)) * np.cos(b * (ax**2))
    kernel_complex.imag = np.exp(-a * (ax**2)) * np.sin(b * (ax**2))
    return kernel_complex.reshape((1, kernel_size))

def normalise_kernels(kernels, params):
    # Normalises with respect to A*real+B*imag
    total = 0

    for k,p in zip(kernels, params):
        # 1D kernel - applied in 2D
        for i in range(k.shape[1]):
            for j in range(k.shape[1]):
                # Complex multiply and weighted sum
                total += p['A'] * (k[0,i].real*k[0,j].real - k[0,i].imag*k[0,j].imag) + p['B'] * (k[0,i].real*k[0,j].imag + k[0,i].imag*k[0,j].real)

    scalar = 1 / math.sqrt(total)
    for kernel in kernels:
        kernel *= scalar

# Combine the real and imaginary parts of an image, weighted by A and B
def weighted_sum(kernel, params):
    return np.add(kernel.real * params['A'], kernel.imag * params['B'])

################################################################################
############ https://github.com/mikepound/convolve/run.lens.py #################
################################################################################

def gamma_exposure(img, gamma):
    return np.power(img, gamma)

def gamma_exposure_inverse(img, gamma):
    img = np.clip(img, 0, None)
    return np.power(img, 1.0/gamma)

def lens_blur(image, radius, n_components, exposure_gamma):
    # Set up an undo group, so the operation will be undone in one step.
    pdb.gimp_image_undo_group_start(image)

    # get active layer
    layer = image.active_layer
    # get the pixels color informations
    img = channelData(layer)
    # reordering the img: [channels x height x width]
    img = np.ascontiguousarray(img.transpose(2,0,1), dtype=np.float32)
    img /= 255

    # Create output of the same size
    output = np.zeros(img.shape, dtype=np.float32)

    # Obtain component parameters / scale values
    parameters, scale = get_parameters(component_count = n_components)
    # Create each component for size radius, using scale and other component parameters
    components = [complex_kernel_1d(radius, scale, component_params['a'], component_params['b']) for component_params in parameters]
    # Normalise all kernels together (the combination of all applied kernels in 2D must sum to 1)
    normalise_kernels(components, parameters)

    # Niklas: pre-multiplication for alpha images
    if img.shape[0] == 4:
        for channel in range(img.shape[0]-1):
            img[channel] *= img[3]


    # Increase exposure to highlight bright spots
    img = gamma_exposure(img, exposure_gamma)

    # NOTE:
    # Let f,g be two complex signals. The convolution f*g can be split as:
    # Re(f)*Re(g) - Im(f)*Im(g) + i [Re(f)*Im(g) + Im(f)*Re(g)]
    # where Re(), Im() represents the real and imaginary parts respectively

    # Process RGB channels for all components
    i = 0.0
    component_output = []
    for component, component_params in zip(components, parameters):
        channels = []
        component_real = np.real(component)
        component_imag = np.imag(component)
        component_real_t = component_real.transpose()
        component_imag_t = component_imag.transpose()
        for channel in range(img.shape[0]):
            # first convolution
            inter_real = cv2.filter2D(img[channel], -1, component_real)
            inter_imag = cv2.filter2D(img[channel], -1, component_imag)
            # second convolution (see NOTE above, here inter_ is f, component_ is g)
            final_1 = cv2.filter2D(inter_real, -1, component_real_t)
            final_2 = cv2.filter2D(inter_real, -1, component_imag_t)
            final_3 = cv2.filter2D(inter_imag, -1, component_real_t)
            final_4 = cv2.filter2D(inter_imag, -1, component_imag_t)
            final = final_1 - final_4 + 1j * (final_2 + final_3)
            channels.append(final)
            # update the progress bar
            i += 1
            pdb.gimp_progress_update(i / (img.shape[0] * len(components)))

        # The final component output is a stack of RGB, with weighted sums of real and imaginary parts
        component_image = np.stack([weighted_sum(channel, component_params) for channel in channels])
        component_output.append(component_image)

    # Add all components together
    output_image = reduce(np.add, component_output)

    # Reverse exposure
    output_image = gamma_exposure_inverse(output_image, exposure_gamma)

    # Niklas inverse pre-multiplication for alpha images
    if output_image.shape[0] == 4:
        for channel in range(output_image.shape[0]-1):
            with np.errstate(divide='ignore', invalid='ignore'):
                output_image[channel] /= output_image[3]

    # Avoid out of range values - generally this only occurs with small negatives
    # due to imperfect complex kernels
    output_image = np.clip(output_image, 0, 1)
    output_image *= 255
    output_image = output_image.transpose(1,2,0).astype(np.uint8)

    # add the new layer to the image
    createResultLayer(image, layer.name + "-bokeh", output_image)

    # Close the undo group.
    pdb.gimp_image_undo_group_end(image)
    # End progress.
    pdb.gimp_progress_end()

register(
        "python_fu_lens_blur",
        "Apply to a target layer the lens blur (also known as bokeh blur).",
        "Apply to a target layer the lens blur (also known as bokeh blur).",
        "Davide Sandona'",
        "Davide Sandona'",
        "2019",
        "Lens Blur...",
        "RGB*, GRAY*",
        [
            (PF_IMAGE, "image", "Input image", None),
            (PF_SLIDER, "radius",  "Radius", 6, (1, 500, 1)),
            (PF_SPINNER, "n_components", "Components", 2, (1, 6, 1)),
            (PF_SPINNER, "exposure_gamma", "Gamma", 3.0, (0, 10, 0.2)),
        ],
        [],
        lens_blur,
        menu="<Image>/Filters/Blur")

main()
