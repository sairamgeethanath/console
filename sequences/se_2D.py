import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import uic
import pickle
import pypulseq as pp  # type: ignore
import external.seq.adjustments_acq.config as cfg
from external.seq.adjustments_acq.scripts import run_pulseq
from sequences.common.get_trajectory import choose_pe_order
from sequences import PulseqSequence
from sequences.common import make_se_2D
from sequences.common import view_traj
import common.logger as logger
from common.types import ResultItem

log = logger.get_logger()


class SequenceSE_2D(PulseqSequence, registry_key=Path(__file__).stem):
    # Sequence parameters
    param_TE: int = 5
    param_TR: int = 1000
    param_NSA: int = 1
    param_FOV: int = 64
    param_Orientation: str = "Coronal"
    param_Base_Resolution: int = 64
    param_BW: int = 16000
    param_Trajectory: str = "Cartesian"
    param_PE_Ordering: str = "Center_out"
    param_PF: int = 1
    param_view_traj: bool = True

    @classmethod
    def get_readable_name(self) -> str:
        return "2D Spin-Echo  [untested]"

    def setup_ui(self, widget) -> bool:
        seq_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(f"{seq_path}/{self.get_name()}/interface.ui", widget)
        return True

    def get_parameters(self) -> dict:
        return {
            "TE": self.param_TE,
            "TR": self.param_TR,
            "NSA": self.param_NSA,
            "FOV": self.param_FOV,
            "Orientation": self.param_Orientation,
            "Base_Resolution": self.param_Base_Resolution,
            "BW": self.param_BW,
            "Trajectory": self.param_Trajectory,
            "PE_Ordering": self.param_PE_Ordering,
            "PF": self.param_PF,
            "view_traj": self.param_view_traj,
        }

    @classmethod
    def get_default_parameters(self) -> dict:
        return {
            "TE": 5,
            "TR": 1000,
            "NSA": 1,
            "FOV": 64,
            "Orientation": "Coronal",
            "Base_Resolution": 64,
            "BW": 16000,
            "Trajectory": "Cartesian",
            "PE_Ordering": "Center_out",
            "PF": 1,
            "view_traj": False,
        }

    def set_parameters(self, parameters, scan_task) -> bool:
        self.problem_list = []
        try:
            self.param_TE = parameters["TE"]
            self.param_TR = parameters["TR"]
            self.param_NSA = parameters["NSA"]
            self.param_FOV = parameters["FOV"]
            self.param_Orientation = parameters["Orientation"]
            self.param_Base_Resolution = parameters["Base_Resolution"]
            self.param_BW = parameters["BW"]
            self.param_Trajectory = parameters["Trajectory"]
            self.param_PE_Ordering = parameters["PE_Ordering"]
            self.param_PF = parameters["PF"]
            self.param_view_traj = parameters["view_traj"]
        except:
            self.problem_list.append("Invalid parameters provided")
            return False
        return self.validate_parameters(scan_task)

    def write_parameters_to_ui(self, widget) -> bool:
        widget.TESpinBox.setValue(self.param_TE)
        widget.TRSpinBox.setValue(self.param_TR)
        widget.NSA_SpinBox.setValue(self.param_NSA)
        widget.Orientation_ComboBox.setCurrentText(self.param_Orientation)
        widget.FOV_SpinBox.setValue(self.param_FOV)
        widget.Base_Resolution_SpinBox.setValue(self.param_Base_Resolution)
        widget.BW_SpinBox.setValue(self.param_BW)
        widget.Trajectory_ComboBox.setCurrentText(self.param_Trajectory)
        widget.PE_Ordering_ComboBox.setCurrentText(self.param_PE_Ordering)
        widget.PF_SpinBox.setValue(self.param_PF)
        widget.visualize_traj_CheckBox.setCheckState(self.param_view_traj)

        return True

    def read_parameters_from_ui(self, widget, scan_task) -> bool:
        self.problem_list = []
        self.param_TE = widget.TESpinBox.value()
        self.param_TR = widget.TRSpinBox.value()
        self.param_NSA = widget.NSA_SpinBox.value()
        self.param_Orientation = widget.Orientation_ComboBox.currentText()
        self.param_FOV = widget.FOV_SpinBox.value()
        self.param_Base_Resolution = widget.Base_Resolution_SpinBox.value()
        self.param_BW = widget.BW_SpinBox.value()
        self.param_Trajectory = widget.Trajectory_ComboBox.currentText()
        self.param_PE_Ordering = widget.PE_Ordering_ComboBox.currentText()
        self.param_PF = widget.PF_SpinBox.value()
        self.param_view_traj = widget.visualize_traj_CheckBox.isChecked()
        self.validate_parameters(scan_task)
        return self.is_valid()

    def validate_parameters(self, scan_task) -> bool:
        if self.param_TE > self.param_TR:
            self.problem_list.append("TE cannot be longer than TR")
        return self.is_valid()

    def calculate_sequence(self, scan_task) -> bool:
        self.seq_file_path = self.get_working_folder() + "/seq/acq0.seq"
        log.info("Calculating sequence " + self.get_name())
        # scan_task.processing.dim = 2
        # scan_task.processing.dim_size = f"{self.param_baseresolution},{2*self.param_baseresolution}"
        # scan_task.processing.oversampling_read = 2
        # scan_task.processing.recon_mode = "basic2d"
        max_grad = np.min([cfg.GX_MAX, cfg.GY_MAX, cfg.GZ_MAX])
        self.system = pp.Opts(
            max_grad=max_grad,  
            grad_unit="Hz/m", # 
            max_slew=1000,
            slew_unit="T/m/s",
            #rf_ringdown_time=100e-6,
            rf_ringdown_time=20e-6,
            rf_dead_time=100e-6,
            rf_raster_time=1e-6,
            #adc_dead_time=10e-6,
            adc_dead_time=20e-6,
            grad_raster_time = 1/self.param_BW,
            B0=0.27,
            )
        log.info("Using system config: ", self.system)
        
        # ToDo: if self.Trajectory == "Cartesian": (default)
        make_se_2D.pypulseq_se2D(
            inputs={
                "TE": self.param_TE,
                "TR": self.param_TR,
                "NSA": self.param_NSA,
                "FOV": self.param_FOV,
                "Orientation": self.param_Orientation,
                "Base_Resolution": self.param_Base_Resolution,
                "BW": self.param_BW,
                "Trajectory": self.param_Trajectory,
                "PE_Ordering": self.param_PE_Ordering,
                "PF": self.param_PF,
                "view_traj": self.param_view_traj,
                "system": self.system,
            },
            check_timing=True,
            output_file=self.seq_file_path,
            output_folder=self.get_working_folder(),
        )
        # elif self.Trajectory == "Radial":
        # pypulseq_se2D_radial(
        #    inputs={"TE": self.param_TE, "TR": self.param_TR}, check_timing=True, output_file=self.seq_file_path
        # )

        log.info("Done calculating sequence " + self.get_name())
        self.calculated = True

        if self.param_view_traj is True:
            log.info("Displaying trajectory... " + self.get_name())
            result = ResultItem()
            result.name = "traj plot"
            result.description = "Plot of trajectory in k space of current sequence."
            result.type = "plot"
            result.primary = True
            result.autoload_viewer = 1
            result.file_path = "other/traj.plot"
            scan_task.results.append(result)

        return True
    
    

    def run_sequence(self, scan_task) -> bool:
        log.info("Running sequence " + self.get_name())

        expected_duration_sec = int(
            self.param_TR
            * (self.param_Base_Resolution)
            / 1000
        )

        rxd, _ = run_pulseq(    
            seq_file=self.seq_file_path,
            rf_center=cfg.LARMOR_FREQ,
            tx_t=1,
            grad_t= np.round(self.system.grad_raster_time * 1e6, decimals=0),
            tx_warmup=100,
            shim_x=cfg.SHIM_X,
            shim_y=cfg.SHIM_Y,
            shim_z=cfg.SHIM_Z,
            grad_cal=False,
            save_np=False,
            save_mat=False,
            save_msgs=False,
            gui_test=False,
            case_path=self.get_working_folder(),
            expected_duration_sec=expected_duration_sec,
            system = self.system,
        )

        # # Compute the average

        rxd_rs = np.reshape(rxd, (2 * self.param_Base_Resolution, self.param_Base_Resolution, self.param_NSA), order='F')
        log.info("New shape of rx data:", rxd_rs.shape)
        rxd_avg = (np.average(rxd_rs, axis=2))
        
        log.info("Done running sequence " + self.get_name())
        # data = rxd_avg.reshape((2 * self.param_Base_Resolution, self.param_Base_Resolution))
        # log.info("Shape of data:", data.shape)
        data = rxd_avg #rxd_avg.reshape((self.param_Base_Resolution, 2 * self.param_Base_Resolution))
        log.info("Plotting figures")
        
        kspace_filter = False

        if kspace_filter is True:
            shape = data.shape
            cutoff_radius_ratio = 0.2
            axes = [np.linspace(-dim/2, dim/2, dim) for dim in shape[0:2]]
            grid = np.meshgrid(*axes, indexing='xy')
            pos = np.stack(grid, axis=-1)
            filter = np.sin(np.pi*np.linalg.norm(pos, axis=-1)/(cutoff_radius_ratio*shape[0]))**2
            filter = 1 - np.transpose(filter)
            log.info(np.max(filter))
            log.info(data.shape)
            data = np.multiply(data, filter)



        plt.clf()
        plt.title(f"k-space data")
        # plt.grid(True, color="#333")
        #log.info("Plotting averaged raw signal")
        plt.imshow(np.abs(data))
        plt.set_cmap('jet')
        plt.clim(0,1.2*np.max(abs(data)))
        file = open(self.get_working_folder() + "/other/kspace.plot", "wb")
        fig = plt.gcf()
        pickle.dump(fig, file)
        file.close()
        result = ResultItem()
        result.name = "k-space"
        result.description = "Acquired k-space"
        result.type = "plot"
        result.autoload_viewer = 1
        result.file_path = "other/kspace.plot"
        scan_task.results.insert(0, result)

        plt.clf()
        plt.title(f"Image data")
        # recon = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data)))
        recon = np.fft.fftshift(np.fft.fft2(data))

        # plt.grid(True, color="#333")
        half_width = int(self.param_Base_Resolution / 2)
        recon2 = np.squeeze(recon[:, half_width:self.param_Base_Resolution + half_width])
        plt.imshow(np.abs(recon))
        plt.set_cmap('gray')
        file = open(self.get_working_folder() + "/other/fft.plot", "wb")
        fig = plt.gcf()
        pickle.dump(fig, file)
        file.close()
        result = ResultItem()
        result.name = "FFT"
        result.description = "FFT of ADC signal"
        result.type = "plot"
        result.autoload_viewer = 2
        result.primary = True
        result.file_path = "other/fft.plot"
        scan_task.results.insert(1, result)

        
        # save the raw data file
        self.raw_file_path = self.get_working_folder() + "/rawdata/raw.npy"
        np.save(self.raw_file_path, data)

        log.info("Saving rawdata, sequence " + self.get_name())
        return True
