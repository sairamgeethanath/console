from pathlib import Path

from external.seq.adjustments_acq.calibration import grad_max_cal
import external.seq.adjustments_acq.config as cfg

import common.logger as logger
import matplotlib.pyplot as plt
import pickle
from common.types import ResultItem
import numpy as np
from sequences import PulseqSequence  # type: ignore
from sequences.common import make_rf_se  # type: ignore
from sequences.common.util import reading_json_parameter, writing_json_parameter

log = logger.get_logger()


# TODO: Untested.
class CalGradAmplitude(PulseqSequence, registry_key=Path(__file__).stem):
    @classmethod
    def get_readable_name(self) -> str:
        return "Calibrate Gradients  [WIP]"

    @classmethod
    def get_description(self) -> str:
        return "Service sequence to calibrate the gradients using a phantom with known dimensions."

    def calculate_sequence(self, scan_task) -> bool:
        scan_task.processing.recon_mode = "bypass"
        self.seq_file_path = self.get_working_folder() + "/seq/acq0.seq"
        log.info("Calculating sequence " + self.get_name())

        make_rf_se.pypulseq_rfse(
            inputs={
                "TE": 3,
                "TR": 250,
                "NSA": 1,
                "ADC_samples": 4096,
                "ADC_duration": 6400,
                "FA1": 90,
                "FA2": 180,
            },
            check_timing=True,
            output_file=self.seq_file_path,
        )

        log.info("Done calculating sequence " + self.get_name())
        self.calculated = True
        return True

    def run_sequence(self, scan_task) -> bool:
        log.info("Running sequence " + self.get_name())

        # reading configuration data from config.json
        configuration_data = reading_json_parameter()

        grad_axes = ["x", "y", "z"]
        iter = 20
        for iterations in range(iter):
            for axis in grad_axes:
                print("test")
                log.info(f"Calibrating {axis} axis")
                grad_max, fft_x, rx_fft, hline = grad_max_cal(
                    channel=axis,
                    phantom_width=30,
                    larmor_freq=cfg.LARMOR_FREQ,
                    calibration_power=0.8,
                    trs=3,
                    tr_spacing=2e6,
                    echo_duration=5000,
                    readout_duration=500,
                    rx_period=25 / 3,
                    RF_PI2_DURATION=100,
                    rf_max=cfg.RF_MAX,
                    trap_ramp_duration=50,
                    trap_ramp_pts=5,
                    plot=False,
                )

            if axis == "x":
                configuration_data.gradients_parameters.gx_maximum = grad_max
            elif axis == "y":
                configuration_data.gradients_parameters.gy_maximum = grad_max
            elif axis == "z":
                configuration_data.gradients_parameters.gz_maximum = grad_max
            writing_json_parameter(config_data=configuration_data)
        plt.clf()
        plt.title(f"Gradient calibration")
        plt.grid(True, color="#333")
        
        
        plt.plot(fft_x, np.abs(rx_fft))
        plt.hlines(*hline, "r")
        # plt.xlabel('Time (us)')
        # plt.ylabel('Signal')
        plt.title(
            f"FFT -- Magnitude ({(grad_max * 1e-3):.4f} KHz/m gradient max)"
        )
        
        file = open(self.get_working_folder() + "/other/gradcal.plot", "wb")
        fig = plt.gcf()
        pickle.dump(fig, file)
        file.close()
        result = ResultItem()
        result.name = "GradCal"
        result.description = "Gradient calibration"
        result.type = "plot"
        result.autoload_viewer = 1
        result.file_path = "other/gradcal.plot"
        scan_task.results.insert(0, result)

        

        log.info("Done running sequence " + self.get_name())





        return True
