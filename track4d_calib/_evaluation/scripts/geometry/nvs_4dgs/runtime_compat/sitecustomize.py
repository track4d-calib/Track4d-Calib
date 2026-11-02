import numpy as np

from track4d_calib.prebuilt_torch_extension import (
    install_prebuilt_extension_from_environment,
)


# Pillow requires unsigned 8-bit RGB arrays from the Fudan reader.
np.byte = np.uint8

install_prebuilt_extension_from_environment()
