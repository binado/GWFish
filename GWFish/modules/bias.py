from typing import Optional

import numpy as np
import pandas as pd

import GWFish.modules.waveforms as wf
from .fishermatrix import FisherMatrix, invertSVD
from .detection import Detector, Network, projection
from .auxiliary import scalar_product


def waveform_bias(
    detector: Detector,
    fiducial_signal_parameter_values: pd.DataFrame,
    fiducial_waveform_model: str,
    recovered_waveform_model: str,
    fisher_parameters: Optional[list[str]] = None,
    f_ref=wf.DEFAULT_F_REF,
    waveform_class: type[wf.Waveform] = wf.LALFD_Waveform,
    long_wavelength: bool = True,
    redefine_tf_vectors: bool = False,
    **kwargs,
):
    data_params = {"frequencyvector": detector.frequencyvector, "f_ref": f_ref}
    fiducial_waveform_generator = waveform_class(
        fiducial_waveform_model, fiducial_signal_parameter_values, data_params
    )
    recovered_waveform_generator = waveform_class(
        recovered_waveform_model, fiducial_signal_parameter_values, data_params
    )
    fiducial_waveform = fiducial_waveform_generator()
    recovered_waveform = recovered_waveform_generator()
    fiducial_t_of_f = fiducial_waveform_generator.t_of_f
    recovered_t_of_f = fiducial_waveform_generator.t_of_f

    if redefine_tf_vectors:
        fiducial_signal, timevector, frequencyvector = projection(
            fiducial_signal_parameter_values,
            detector,
            fiducial_waveform,
            fiducial_t_of_f,
            redefine_tf_vectors=True,
            long_wavelength_approx=long_wavelength,
        )
        recovered_signal, timevector, frequencyvector = projection(
            fiducial_signal_parameter_values,
            detector,
            recovered_waveform,
            recovered_t_of_f,
            redefine_tf_vectors=True,
            long_wavelength_approx=long_wavelength,
        )

    else:
        fiducial_signal = projection(
            fiducial_signal_parameter_values,
            detector,
            fiducial_waveform,
            fiducial_t_of_f,
            long_wavelength_approx=long_wavelength,
        )
        recovered_signal = projection(
            fiducial_signal_parameter_values,
            detector,
            recovered_waveform,
            recovered_t_of_f,
            long_wavelength_approx=long_wavelength,
        )

    delta_h = fiducial_signal - recovered_signal  # type: ignore

    if fisher_parameters is None:
        if isinstance(fiducial_signal_parameter_values, dict):
            fisher_parameters = list(fiducial_signal_parameter_values.keys())
        else:
            fisher_parameters = fiducial_signal_parameter_values.columns

    fisher_matrix = FisherMatrix(
        fiducial_waveform_model,
        fiducial_signal_parameter_values,
        fisher_parameters,
        detector,
        waveform_class=waveform_class,
        f_ref=f_ref,
        **kwargs,
    )
    covariance_matrix, _ = invertSVD(fisher_matrix.fm)
    num_parameters = len(fisher_parameters)
    projection_on_residuals = np.empty((num_parameters,))
    for i in range(num_parameters):
        projection_on_residuals[i] = scalar_product(delta_h, fisher_matrix.derivative_array[i, ...], detector).sum()
    return covariance_matrix @ projection_on_residuals
