import bilby
import numpy as np
from celerite.modeling import Model
import lalsimulation as lalsim
import lal
from scipy.interpolate import interp1d
from bilby.gw.utils import _get_lalsim_approximant, convert_args_list_to_float
from bilby.gw.detector.interferometer import get_polarization_tensor
from bilby.core.utils.constants import solar_mass, parsec


def get_model(inputs):
    return GWModel


class _Model(Model):
    @property
    def parameters(self):
        return {key: getattr(self, key) for key in self.parameter_names}


class GWModel(_Model):
    def __init__(self, **init_values):
        self.parameter_names = tuple(key for key in init_values)
        super().__init__(**init_values)
        self.bilby_detector = None
        self.pre_trigger_duration = None
        self.bilby_waveform_generator = None

    def initiate_model(self, strain, inputs):
        self.strain = strain
        self.inputs = inputs

        waveform_dictionary = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(
            waveform_dictionary, int(inputs.pn_amplitude_order)
        )
        self.waveform_dictionary = waveform_dictionary

        if inputs.mode_array is not None:
            mode_array_lal = lalsim.SimInspiralCreateModeArray()
            for mode in inputs.mode_array:
                lalsim.SimInspiralModeArrayActivateMode(
                    mode_array_lal, mode[0], mode[1]
                )
            lalsim.SimInspiralWaveformParamsInsertModeArray(
                waveform_dictionary, mode_array_lal
            )

        self.bilby_detector = bilby.gw.detector.get_empty_interferometer(
            strain.detector
        )
        self.pre_trigger_duration = inputs.duration - inputs.post_trigger_duration

    def get_value(self, time):
        # Coudld add a check here
        time = self.strain.xunprocessed
        waveform = self.get_gw_waveform(time)
        _, waveform_processed = self.strain.process(time, waveform)
        return waveform_processed

    def get_gw_waveform(self, time):

        return get_gw_waveform(
            time,
            self.parameters,
            self.inputs.waveform_approximant,
            self.inputs.reference_frequency,
            self.bilby_detector,
            fudge=0.80,
            reference_frame=self.inputs.whittle_likelihood.reference_frame,
            pre_trigger_duration=self.pre_trigger_duration,
            error=False,
            waveform_dictionary=self.waveform_dictionary,
            waveform_generator=self.get_bilby_waveform_generator(time),
            frequency_domain_source_model=self.inputs.frequency_domain_source_model,
        )

    def get_bilby_waveform_generator(self, time):
        if self.bilby_waveform_generator is None:
            self.bilby_waveform_generator = instantiate_bilby_waveform_generator(
                self.inputs, time
            )
        return self.bilby_waveform_generator


def get_gw_waveform(
    time,
    parameters,
    waveform_approximant,
    reference_frequency,
    bilby_detector,
    fudge=0.70,
    reference_frame=None,
    pre_trigger_duration=None,
    error=False,
    waveform_dictionary=None,
    waveform_generator=None,
    frequency_domain_source_model=None,
):
    par, _ = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(parameters)

    mass_1_SI = par["mass_1"] * solar_mass
    mass_2_SI = par["mass_2"] * solar_mass
    luminosity_distance_SI = par["luminosity_distance"] * 1e6 * parsec

    # Extract information about the time series
    deltaT = time[1] - time[0]
    duration = time[-1] - time[0]
    if pre_trigger_duration is None:
        nearest_trigger_idx = np.argmin(np.abs(time - par["geocent_time"]))
        pre_trigger_duration = time[nearest_trigger_idx] - time[0]

    # Get the approximant number from the name
    approximant = _get_lalsim_approximant(waveform_approximant)

    # Estimate a minimum frequency required to ensure the waveform covers the data
    # Note there is a fudge factor as SimInspiralChirpStartFrequencyBound includes
    # only the leading order Newtonian coefficient.

    if "NRSur" in waveform_approximant:
        f_min = 0
    else:
        f_min = fudge * lalsim.SimInspiralChirpStartFrequencyBound(
            pre_trigger_duration,
            mass_1_SI,
            mass_2_SI,
        )

    # Check if the reference frequency is used, if not use f_min
    if (
        lalsim.SimInspiralGetSpinFreqFromApproximant(approximant)
        == lalsim.SIM_INSPIRAL_SPINS_FLOW
    ):
        f_ref = f_min
    elif reference_frequency == "fmin":
        f_ref = f_min
    else:
        f_ref = reference_frequency

    (
        iota,
        spin_1x,
        spin_1y,
        spin_1z,
        spin_2x,
        spin_2y,
        spin_2z,
    ) = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        theta_jn=par["theta_jn"],
        phi_jl=par["phi_jl"],
        tilt_1=par["tilt_1"],
        tilt_2=par["tilt_2"],
        phi_12=par["phi_12"],
        a_1=par["a_1"],
        a_2=par["a_2"],
        mass_1=mass_1_SI,
        mass_2=mass_2_SI,
        reference_frequency=f_ref,
        phase=par["phase"],
    )

    if "zenith" in par and "azimuth" in par:
        par["ra"], par["dec"] = bilby.gw.utils.zenith_azimuth_to_ra_dec(
            par["zenith"], par["azimuth"], par["geocent_time"], reference_frame
        )

    longitude_ascending_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0

    args = convert_args_list_to_float(
        mass_1_SI,
        mass_2_SI,
        spin_1x,
        spin_1y,
        spin_1z,
        spin_2x,
        spin_2y,
        spin_2z,
        luminosity_distance_SI,
        iota,
        par["phase"],
        longitude_ascending_nodes,
        eccentricity,
        mean_per_ano,
        deltaT,
        f_min,
        f_ref,
    )

    if waveform_dictionary is None:
        waveform_dictionary = lal.CreateDict()

    use_bilby_generator = should_use_bilby_waveform_generator(
        approximant, frequency_domain_source_model
    )

    if use_bilby_generator:
        if waveform_generator is None:
            raise ValueError(
                f"No bilby waveform generator available for {waveform_approximant}"
            )
        strain_interp = get_gw_waveform_via_bilby_generator(
            time=time,
            parameters=parameters,
            sky_parameters=par,
            bilby_detector=bilby_detector,
            waveform_generator=waveform_generator,
        )
    else:
        try:
            h_plus_timeseries, h_cross_timeseries = lalsim.SimInspiralChooseTDWaveform(
                *args, waveform_dictionary, approximant
            )
        except Exception as exc:
            if waveform_generator is None or not should_retry_with_bilby_waveform_generator(
                exc
            ):
                raise
            strain_interp = get_gw_waveform_via_bilby_generator(
                time=time,
                parameters=parameters,
                sky_parameters=par,
                bilby_detector=bilby_detector,
                waveform_generator=waveform_generator,
            )
        else:
            plus_polarization_tensor = get_polarization_tensor(
                par["ra"], par["dec"], par["geocent_time"], par["psi"], "plus"
            )
            f_plus = np.einsum(
                "ij,ij->", bilby_detector.detector_tensor, plus_polarization_tensor
            )

            cross_polarization_tensor = get_polarization_tensor(
                par["ra"], par["dec"], par["geocent_time"], par["psi"], "cross"
            )
            f_cross = np.einsum(
                "ij,ij->", bilby_detector.detector_tensor, cross_polarization_tensor
            )

            h_plus = h_plus_timeseries.data.data
            h_cross = h_cross_timeseries.data.data
            h_plus_time = np.arange(len(h_plus)) * h_plus_timeseries.deltaT + float(
                h_plus_timeseries.epoch
            )

            time_shift = bilby_detector.time_delay_from_geocenter(
                par["ra"], par["dec"], par["geocent_time"]
            )

            predicted_strain = f_plus * h_plus + f_cross * h_cross
            predicted_strain_time = h_plus_time + par["geocent_time"] + time_shift

            strain_interp = interp1d(
                predicted_strain_time,
                predicted_strain,
                fill_value=0,
                bounds_error=False,
            )(time)

    if strain_interp[0] == 0:
        idxs = strain_interp != 0
        fduration = time[idxs][0] - time[0]
        frac = fduration / duration
        msg = f"Generated waveform too short: no signal for the first {fduration:1.2f}s ({frac:1.2f} of total)"
        if error:
            raise ValueError(msg)
        else:
            if fduration > 0.9:
                pass
    return strain_interp


def instantiate_bilby_waveform_generator(inputs, time):
    delta_t = float(time[1] - time[0])
    duration = float(len(time) * delta_t)
    constructor_kwargs = inputs.get_default_waveform_generator_class_ctor_arguments()
    for key in [
        "duration",
        "sampling_frequency",
        "start_time",
        "frequency_domain_source_model",
        "time_domain_source_model",
        "parameter_conversion",
        "waveform_arguments",
    ]:
        constructor_kwargs.pop(key, None)

    return inputs.waveform_generator_class(
        **constructor_kwargs,
        frequency_domain_source_model=inputs.bilby_frequency_domain_source_model,
        sampling_frequency=1.0 / delta_t,
        duration=duration,
        start_time=float(time[0]),
        parameter_conversion=inputs.parameter_conversion,
        waveform_arguments=inputs.get_default_waveform_arguments(),
    )


def should_use_bilby_waveform_generator(approximant, frequency_domain_source_model):
    uses_custom_source_model = frequency_domain_source_model not in [
        None,
        "lal_binary_black_hole",
    ]
    return uses_custom_source_model or not lalsim.SimInspiralImplementedTDApproximants(
        approximant
    )


def should_retry_with_bilby_waveform_generator(exc):
    message = " ".join(str(arg) for arg in exc.args)
    return "no generator defined" in message.lower() or "check failed: generator" in message.lower()


def get_gw_waveform_via_bilby_generator(
    time, parameters, sky_parameters, bilby_detector, waveform_generator
):
    try:
        waveform_polarizations = waveform_generator.time_domain_strain(parameters)
    except TypeError as exc:
        # bilby_tgr.pseob returns None for handled waveform-domain failures.
        if "'NoneType' object is not iterable" in str(exc):
            raise ValueError(
                "Waveform generator returned no time-domain polarizations"
            ) from exc
        raise

    if waveform_polarizations is None:
        raise ValueError("Waveform generator returned no time-domain polarizations")

    signal = np.zeros(len(time))
    for mode, polarization in waveform_polarizations.items():
        det_response = bilby_detector.antenna_response(
            sky_parameters["ra"],
            sky_parameters["dec"],
            sky_parameters["geocent_time"],
            sky_parameters["psi"],
            mode,
        )
        signal += np.asarray(polarization) * det_response

    time_shift = bilby_detector.time_delay_from_geocenter(
        sky_parameters["ra"], sky_parameters["dec"], sky_parameters["geocent_time"]
    )
    dt = sky_parameters["geocent_time"] + time_shift - waveform_generator.start_time
    n_roll = int(np.round(dt * waveform_generator.sampling_frequency))

    return np.roll(signal, n_roll)
