import bilby
import bilby_pipe
from bilby_pipe.data_analysis import DataAnalysisInput
from bilby_pipe.utils import nonestr, convert_string_to_list, convert_prior_string_input, convert_string_to_dict


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Unable to parse boolean value {value!r}")


class GPBilbyInputs(DataAnalysisInput):
    """Handles user-input for the GPBilbyInputs script"""

    def __init__(self, args, unknown_args):
        super().__init__(args, unknown_args)

        self.cleaning = args.cleaning
        self.trigger_time = args.trigger_time

        self.duration = args.duration
        self.post_trigger_duration = args.post_trigger_duration

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

        self.noise_only_run = args.noise_only_run

        # GP
        kernel_prior_dict = convert_prior_string_input(args.kernel_prior_dict)
        new_kernel_prior_dict = {}
        for key in kernel_prior_dict:
            new_kernel_prior_dict[key.replace("-", "_")] = kernel_prior_dict[key]
        self.kernel_prior_dict = bilby.core.prior.ConditionalPriorDict(
            new_kernel_prior_dict
        )

        self.kernel_list = sorted([key.lower() for key in args.kernel])

        # Outputs
        self.plot_xmin = args.plot_xmin
        self.plot_xmax = args.plot_xmax
        self.plot_ymin = args.plot_ymin
        self.plot_ymax = args.plot_ymax

        # Likelihood and prior
        self.whittle_likelihood, self.prior = self.get_likelihood_and_priors()
        self.default_prior = args.default_prior
        self.yerr_scaled = args.yerr_scaled

        self.data_cuts = convert_string_to_list(args.data_cuts)

        # Calibration
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict
        self.convert_calibration = args.convert_calibration


def create_parser():
    parser = bilby_pipe.main.create_parser(top_level=False)
    gp = parser.add_argument_group("Gaussian-Process Arguments")
    gp.add(
        "--cleaning",
        type=str,
        default="whiten",
    )
    gp.add(
        "--data-cuts",
        type=nonestr,
        action="append",
    )
    gp.add(
        "--plot-xmin",
        type=float,
        default=None,
    )
    gp.add(
        "--plot-xmax",
        type=float,
        default=None,
    )
    gp.add(
        "--plot-ymin",
        type=float,
        default=None,
    )
    gp.add(
        "--plot-ymax",
        type=float,
        default=None,
    )
    gp.add(
        "--kernel",
        type=nonestr,
        default=None,
        action="append",
        help="A list of kernels",
    )
    gp.add(
        "--kernel-prior-dict",
        type=nonestr,
        default=None,
        help="A dictionary of kernel priors (alternative to prior-file)",
    )
    gp.add("--noise-only-run", default=False, action="store_true", help="")
    gp.add(
        "--yerr-scaled",
        type=float,
        default=1e-2,
        help="Fractional amplitude uncertainty",
    )
    gp.add(
        "--convert-calibration",
        type=parse_bool,
        default=True,
        help="If true, convert calibration uncertainty"
    )

    return parser
