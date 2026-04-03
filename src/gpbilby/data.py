import os

import numpy as np
from gwpy.signal import filter_design
from gwpy.timeseries import TimeSeries
from bilby.core.utils import nfft, infft
import lal
import gwpy

from .lines import LineList
from .gpmodel import get_model
from .calibration import measure_yerr_from_calibration_envelope


def get_data_dictionary(inputs):
    data_dictionary = get_raw_data_dictionary(inputs)
    return data_dictionary


def get_raw_data_dictionary(inputs):
    data_dictionary = {}
    for ifo in inputs.interferometers:
        strain = Strain(ifo, inputs)
        data_dictionary[ifo.name] = strain

    return data_dictionary


def inject_signal(data_dictionary, inputs):
    init_parameters = inputs.priors.sample()

    print("Injecting signal into data")
    for detector, strain in data_dictionary.items():
        model = get_model(inputs)(**init_parameters)
        model.strain = strain
        model.inputs = inputs

        for key, val in inputs.injection_dict.items():
            model.set_parameter(key, val)

        pred = model.get_value(strain.x_shifted)
        strain.y += pred * strain.scale
        strain.y_scaled += pred
    return data_dictionary


class Strain(object):
    def __init__(self, ifo, inputs):
        self.asd = ifo.amplitude_spectral_density_array
        self.frequency_array = ifo.frequency_array
        self.frequency_domain_strain = ifo.frequency_domain_strain
        self.time_domain_strain = ifo.time_domain_strain
        self.time_array = ifo.time_array

        self.data_cuts = inputs.data_cuts

        self.trigger_time = inputs.trigger_time

        self.inputs = inputs
        self.detector = ifo.name

        self.window = ifo.strain_data.time_domain_window()

        self.xunprocessed = self.time_array
        self.yunprocessed = self.time_domain_strain

        self.init_idxs()

        self.x, self.y = self.process(self.xunprocessed, self.yunprocessed)

        self.scale = 1
        self.y_scaled = self.y / self.scale
        self.x_shifted = self.x - self.trigger_time
        self.xunprocessed_shifted = self.xunprocessed - self.trigger_time

        self.process_calibration(ifo, inputs)

    def process_calibration(self, ifo, inputs):
        if inputs.convert_calibration:
            self.yerr = measure_yerr_from_calibration_envelope(
                frequencies=self.frequency_array,
                h_f=self.frequency_domain_strain,
                psd=ifo.power_spectral_density_array,
                calibration_envelope=inputs.spline_calibration_envelope_dict[ifo.name],
                nsim=10,
                min_frequency=ifo.minimum_frequency,
                max_frequency=ifo.maximum_frequency,
            )
            self.yerr_scaled = self.yerr / self.scale
            self.yerr_scaled = self.yerr_scaled[self.idxs]
        else:
            raw_yerr_scaled = np.asarray(inputs.yerr_scaled, dtype=float)
            if raw_yerr_scaled.ndim == 0:
                self.yerr_scaled = np.full_like(self.x, raw_yerr_scaled.item(), dtype=float)
            elif raw_yerr_scaled.shape == self.x.shape:
                self.yerr_scaled = raw_yerr_scaled
            else:
                self.yerr_scaled = raw_yerr_scaled[self.idxs]

            print(f"WARNING: using yerr of {self.yerr_scaled}")
            self.yerr = self.yerr_scaled * self.scale

    def process(self, x, y):
        y = y * self.window
        if "whiten" in self.inputs.cleaning:
            y = apply_whiten(
                y, self.inputs.sampling_frequency, self.inputs.duration, self.asd
            )
        if "highpass" in self.inputs.cleaning:
            y = apply_highpass(x, y, self.inputs.minimum_frequency)
        y = y[self.idxs]
        x = x[self.idxs]
        return x, y

    @property
    def color(self):
        lookup = dict(H1="C7", L1="C7", V1="C7")
        return lookup[self.detector]

    def init_idxs(self):
        self.idxs = np.full(len(self.xunprocessed), True)
        for cut in self.data_cuts:
            if cut is None:
                continue

            cmin, cmax = cut.rstrip(")").lstrip("(").split("|")
            cmin = cmin.replace(" ", "")
            cmax = cmax.replace(" ", "")
            if cmin in ["start", "s"]:
                cmin = self.time_array[0]
            else:
                cmin = float(cmin)
                cmin += self.trigger_time
            if cmax in ["end", "e"]:
                cmax = self.time_array[-1]
            else:
                cmax = float(cmax)
                cmax += self.trigger_time

            print(f"Cutting data from {cmin} to {cmax}")

            self.idxs *= (self.xunprocessed < cmin) | (self.xunprocessed > cmax)


def apply_whiten(y, sampling_frequency, duration, asd):
    yf, f = nfft(y, sampling_frequency)
    ytw = infft(
        yf * np.sqrt(2.0 / sampling_frequency / duration) / asd, sampling_frequency
    )
    return ytw


def apply_highpass(x, y, minimum_frequency):
    data = TimeSeries(y, times=x)
    hp = filter_design.highpass(minimum_frequency, data.sample_rate)
    zpk = filter_design.concatenate_zpks(hp)
    return data.filter(zpk, filtfilt=True).value


def apply_filter_and_notch(x, y, inputs):
    data = TimeSeries(y, times=x)
    bp_dict = dict(
        flow=inputs.minimum_frequency,
        fhigh=inputs.maximum_frequency,
        sample_rate=data.sample_rate,
    )

    # Set up bandpass
    bp = filter_design.bandpass(**bp_dict)

    # Apply
    zpk = filter_design.concatenate_zpks(bp)

    return data.filter(zpk, filtfilt=True).value


def OLD_apply_filter_and_notch(data, inputs):
    detector = data.name.split(":")[0]
    bp_dict = dict(
        flow=inputs.minimum_frequency,
        fhigh=inputs.maximum_frequency,
        sample_rate=data.sample_rate,
    )

    notch_list = [
        notch for notch in inputs.notch_list if notch < data.sample_rate.value / 2
    ]

    if inputs.lines_file_dict and detector in inputs.lines_file_dict:
        lines_list = LineList(
            inputs.lines_file_dict[detector],
            inputs.minimum_frequency,
            inputs.maximum_frequency,
        )
        notch_list += lines_list.get_central_frequency_list()

    notch_list = sorted(notch_list)

    print(
        f"{detector}: applying bandpass filter with {bp_dict} and notches={notch_list}"
    )

    # Set up bandpass
    bp = filter_design.bandpass(**bp_dict)

    # Set up notches
    notches = [filter_design.notch(line, data.sample_rate) for line in notch_list]

    # Apply
    zpk = filter_design.concatenate_zpks(bp, *notches)

    return data.filter(zpk, filtfilt=True)


def get_strain_data(ifo, start, end, inputs, sampling_frequency=512):

    print(f"Getting data for {ifo} from {start} to {end}")
    channel = inputs.channel_dict[ifo]
    if inputs.data_dict is not None:
        source = inputs.data_dict[ifo]
    else:
        source = None

    if channel == "GWOSC":
        data = TimeSeries.fetch_open_data(ifo, start, end, cache=True, verbose=True)
        data.name = f"{ifo}:GWOSC"
    elif source is not None:
        if os.path.isfile(source):
            full_channel = "{}:{}".format(ifo, channel)
            data = TimeSeries.read(source, full_channel, start, end)
        else:
            raise ValueError(f"Source file {source} not found")
    else:
        full_channel = "{}:{}".format(ifo, channel)
        print(f"Running read {channel} from {start} to {end}")
        data = TimeSeries.get(full_channel, start, end)

    if data.sample_rate != sampling_frequency:
        if inputs.resampling_method == "lal":
            lal_timeseries = data.to_lal()
            lal.ResampleREAL8TimeSeries(
                lal_timeseries, float(1 / inputs.sampling_frequency)
            )
            data = gwpy.timeseries.TimeSeries(
                lal_timeseries.data.data,
                epoch=lal_timeseries.epoch,
                dt=lal_timeseries.deltaT,
                name=full_channel,
            )
        else:
            data = data.resample(sampling_frequency)

    return data


def get_per_ifo_timeseries(ifo, inputs):
    start_time = inputs.start_time
    end_time = inputs.start_time + inputs.duration
    sampling_frequency = inputs.sampling_frequency
    timeseries = get_strain_data(
        ifo, start_time, end_time, inputs=inputs, sampling_frequency=sampling_frequency
    )
    return timeseries
