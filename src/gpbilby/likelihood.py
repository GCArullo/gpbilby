import bilby
import celerite
from celerite import terms
import numpy as np

from .gpmodel import get_model

KNOWN_WAVEFORM_FAILURE_MESSAGES = (
    "Waveform generator returned no time-domain polarizations",
)
KNOWN_GP_SOLVER_FAILURE_MESSAGES = (
    "failed to factorize or solve matrix",
)
MAX_GP_SOLVER_DIAGNOSTICS = 5


class FrequencySHOTerm(terms.SHOTerm):
    parameter_names = ("log_S0", "log_Q", "frequency")

    def __repr__(self):
        return "SHOTerm({0.log_S0}, {0.log_Q}, {0.frequency})".format(self)

    def get_real_coefficients(self, params):
        log_S0, log_Q, frequency = params
        log_omega0 = np.log(2 * np.pi * frequency)
        Q = np.exp(log_Q)
        if Q >= 0.5:
            return np.empty(0), np.empty(0)

        S0 = np.exp(log_S0)
        w0 = np.exp(log_omega0)
        f = np.sqrt(1.0 - 4.0 * Q**2)
        return (
            0.5*S0*w0*Q*np.array([1.0+1.0/f, 1.0-1.0/f]),
            0.5*w0/Q*np.array([1.0-f, 1.0+f])
        )

    def get_complex_coefficients(self, params):
        log_S0, log_Q, frequency = params
        log_omega0 = np.log(2 * np.pi * frequency)
        Q = np.exp(log_Q)
        if Q < 0.5:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        S0 = np.exp(log_S0)
        w0 = np.exp(log_omega0)
        f = np.sqrt(4.0 * Q**2-1)
        return (
            S0 * w0 * Q,
            S0 * w0 * Q / f,
            0.5 * w0 / Q,
            0.5 * w0 / Q * f,
        )


def get_likelihood(data_dictionary, inputs, priors, **kwargs):
    init_parameters = priors.sample()
    model_init_parameters = {
        key: val for key, val in init_parameters.items() if "kernel" not in key
    }

    model_dictionary = {}
    for detector, strain in data_dictionary.items():
        model = get_model(inputs)(**model_init_parameters)
        model.initiate_model(strain=strain, inputs=inputs)
        model_dictionary[detector] = model

    return MultiDetectorLikelihood(model_dictionary, inputs, **kwargs)


class MultiDetectorLikelihood(bilby.Likelihood):
    def __init__(
        self, model_dictionary, inputs, signal=True, log_noise_evidence=np.nan
    ):
        self.model_dictionary = model_dictionary
        self.likelihood_dictionary = {}
        self.detector_parameter_names = {}
        self._noise_log_likelihood = log_noise_evidence

        parameters = {}
        for detector, model in self.model_dictionary.items():

            # Set up Bilby likelihood
            likelihood = SingleDetectorCeleriteLikelihood(
                model, signal, inputs.kernel_list
            )
            self.likelihood_dictionary[detector] = likelihood

            # Add keys to global parameters, provide unique kernel names for each detector
            keys = [key for key in likelihood.parameters]
            self.detector_parameter_names[detector] = set(keys)
            parameters.update({key: None for key in keys})

        super().__init__(parameters=parameters)

    def log_likelihood(self, parameters=None):
        if parameters is None:
            parameters = self.parameters

        logl = 0
        for detector, likelihood in self.likelihood_dictionary.items():
            likelihood_parameters = {}
            detector_parameter_names = self.detector_parameter_names[detector]

            # Update the parameters of the per-detector likelihoods
            for key, value in parameters.items():
                if key in detector_parameter_names:
                    likelihood_parameters[key] = value

                lkey = key.replace(f"{detector}-", "")
                if lkey in detector_parameter_names:
                    likelihood_parameters[lkey] = value

            logl += likelihood.log_likelihood(parameters=likelihood_parameters)

        return logl

    def noise_log_likelihood(self):
        return self._noise_log_likelihood


class SingleDetectorCeleriteLikelihood(bilby.Likelihood):
    def __init__(self, model, signal, kernel_list):
        self.x = model.strain.x
        self.y_scaled = model.strain.y_scaled
        self.yerr_scaled = model.strain.yerr_scaled
        self.detector = model.strain.detector

        # Set up the GP model
        kernels = []
        for kernel_name in kernel_list:
            if kernel_name == "matern32term":
                kernels.append(terms.Matern32Term(log_sigma=0, log_rho=0, eps=1e-2))
            elif kernel_name == "fshoterm":
                kernels.append(FrequencySHOTerm(log_S0=0, log_Q=0, frequency=10))
            elif kernel_name == "jitterterm":
                kernels.append(terms.JitterTerm(log_sigma=0))
            elif kernel_name == "realterm":
                kernels.append(terms.RealTerm(log_a=0, log_c=0))
            else:
                raise ValueError(f"Kernel name {kernel_name} unknown")

        kernel = kernels[0]
        for kern in kernels[1:]:
            kernel += kern

        if signal is False:
            model = 0
        self.gp = celerite.GP(kernel, mean=model, fit_mean=signal)
        self.gp.compute(self.x, yerr=self.yerr_scaled)
        self.gp_parameter_names = self.gp.parameter_names
        self._gp_solver_diagnostic_count = 0

        parameters = {}
        for name in self.gp_parameter_names:
            if "kernel" in name:
                bname = name.replace("kernel", f"{self.detector}-kernel")
                bval = None
            elif "value" in name:
                bname = name.replace("mean:", "")
                bval = 0
            elif "mean" in name:
                bname = name.replace("mean:", "")
                bval = None

            parameters[bname] = bval

        super().__init__(parameters=parameters)

    @staticmethod
    def _format_parameter_items(parameter_dict):
        items = []
        for key, value in sorted(parameter_dict.items()):
            if isinstance(value, np.generic):
                value = value.item()

            if isinstance(value, float):
                items.append(f"{key}={value:.6e}")
            else:
                items.append(f"{key}={value}")

        return ", ".join(items)

    @staticmethod
    def _summarize_array(name, values):
        values = np.asarray(values)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return f"{name}: size={values.size}, finite=0"

        return (
            f"{name}: size={values.size}, finite={finite_values.size}, "
            f"min={np.min(finite_values):.6e}, "
            f"median={np.median(finite_values):.6e}, "
            f"max={np.max(finite_values):.6e}"
        )

    def _summarize_mean_model(self):
        mean_model = getattr(self.gp, "mean", None)
        if mean_model is None:
            return "mean_model: unavailable"

        if hasattr(mean_model, "get_value"):
            try:
                mean_value = mean_model.get_value(self.x)
            except Exception as exc:
                return f"mean_model: unavailable ({exc})"
            return self._summarize_array("mean_model", mean_value)

        return f"mean_model: scalar={mean_model}"

    def _emit_gp_solver_diagnostics(self, exception, parameters):
        self._gp_solver_diagnostic_count += 1
        if self._gp_solver_diagnostic_count > MAX_GP_SOLVER_DIAGNOSTICS:
            if self._gp_solver_diagnostic_count == MAX_GP_SOLVER_DIAGNOSTICS + 1:
                print(
                    f"GP solver diagnostics suppressed for {self.detector} after "
                    f"{MAX_GP_SOLVER_DIAGNOSTICS} failures"
                )
            return

        gp_parameters = self.gp.get_parameter_dict()
        kernel_parameters = {
            key: value for key, value in gp_parameters.items() if key.startswith("kernel")
        }
        mean_parameters = {
            key: value for key, value in gp_parameters.items() if key.startswith("mean:")
        }

        print(
            f"GP solver diagnostic [{self.detector} "
            f"{self._gp_solver_diagnostic_count}/{MAX_GP_SOLVER_DIAGNOSTICS}]: {exception}"
        )
        print(
            "  kernel parameters: "
            f"{self._format_parameter_items(kernel_parameters)}"
        )
        if mean_parameters:
            print(
                "  mean parameters: "
                f"{self._format_parameter_items(mean_parameters)}"
            )
        print(
            "  likelihood parameters: "
            f"{self._format_parameter_items(parameters)}"
        )
        print(f"  {self._summarize_array('y_scaled', self.y_scaled)}")
        print(f"  {self._summarize_array('yerr_scaled', self.yerr_scaled)}")
        print(f"  {self._summarize_mean_model()}")

    def update_gp_parameters(self, parameters):
        for key, val in parameters.items():
            if "kernel" in key:
                if self.detector in key:
                    gpkey = key.replace(f"{self.detector}-", "")
                    self.gp.set_parameter(gpkey, val)
            elif "value" in key:
                gpkey = f"mean:{key}"
                self.gp.set_parameter(gpkey, 0)
            else:
                gpkey = f"mean:{key}"
                if gpkey in self.gp_parameter_names:
                    self.gp.set_parameter(gpkey, val)

    def log_likelihood(self, parameters=None):
        if parameters is None:
            parameters = self.parameters

        self.update_gp_parameters(parameters)
        try:
            return self.gp.log_likelihood(self.y_scaled)
        except Exception as e:
            exception_message = str(e)
            if any(message in exception_message for message in KNOWN_WAVEFORM_FAILURE_MESSAGES):
                return -np.inf
            if any(
                message in exception_message.lower()
                for message in KNOWN_GP_SOLVER_FAILURE_MESSAGES
            ):
                print(f"Likelihood evaluation failed: {e}")
                self._emit_gp_solver_diagnostics(e, parameters)
                return -np.inf
            print(f"Likelihood evaluation failed: {e}")
            return -np.inf

    def predict_signal(self, x, sample):
        self.update_gp_parameters(sample)
        return self.gp.mean.get_value(x)

    def predict(self, x, sample):
        self.update_gp_parameters(sample)
        return self.gp.predict(self.y_scaled, x, return_var=True)
