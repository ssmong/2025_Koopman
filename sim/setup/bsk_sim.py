import os
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros, unitTestSupport, vizSupport
from sim.setup.dynamics import Dynamics
from sim.setup.environment import Environment
from sim.setup.fsw import FSW
from sim.utils.plot import (
    plot_attitude_error,
    plot_rate_error,
    plot_torque,
    plot_rw_motor_torque,
    plot_rw_speeds,
)

class BskSim:
    def __init__(self, sim_cfg, ctl_cfg):
        self.sim_cfg = sim_cfg
        self.ctl_cfg = ctl_cfg
        self._setup_sim()
        
        self.dyn = Dynamics(self)
        self.env = Environment(self)
        self.fsw = FSW(self)
        
        self._setup_logging()
        self._setup_viz()
        
    def _setup_sim(self):
        self.simTaskName = "simTask"
        self.fswTaskName = "fswTask"
        simProcessName = "simProcess"

        self.scSim = SimulationBaseClass.SimBaseClass()
        self.scSim.SetProgressBar(False)

        dynProcess = self.scSim.CreateNewProcess(simProcessName)

        self.simTimeNano = macros.sec2nano(self.sim_cfg.sim_time)
        self.simDtNano = macros.sec2nano(self.sim_cfg.sim_dt)
        self.ctrlDtNano = macros.sec2nano(self.sim_cfg.ctrl_dt)
        
        dynProcess.addTask(self.scSim.CreateNewTask(self.simTaskName, self.simDtNano))
        dynProcess.addTask(self.scSim.CreateNewTask(self.fswTaskName, self.ctrlDtNano))

    def _setup_logging(self):
        samplingTime = macros.sec2nano(self.sim_cfg.log_dt)
        
        self.attErrorLog = self.fsw.attError.attGuidOutMsg.recorder(samplingTime)
        self.torqueLog = self.fsw.controller.cmdTorqueOutMsg.recorder(samplingTime)
        
        self.scSim.AddModelToTask(self.simTaskName, self.attErrorLog)
        self.scSim.AddModelToTask(self.simTaskName, self.torqueLog)
        
        if self.sim_cfg.sc.actuators.use_rw:
            self.rwMotorTorqueLog = self.dyn.rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
            self.rwStateLog = self.dyn.rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
            self.scSim.AddModelToTask(self.simTaskName, self.rwMotorTorqueLog)
            self.scSim.AddModelToTask(self.simTaskName, self.rwStateLog)

    def _setup_viz(self):
        if self.sim_cfg.use_vizard and vizSupport.vizFound:
            rw_effector_list = []
            if self.sim_cfg.sc.actuators.use_rw:
                rw_effector_list = [self.dyn.rwStateEffector]

            self.viz = vizSupport.enableUnityVisualization(
                self.scSim, self.simTaskName, self.dyn.scObject,
                rwEffectorList=rw_effector_list,
                liveStream=True
            )

    def init_simulation(self):
        self.scSim.InitializeSimulation()
        self.scSim.ConfigureStopTime(self.simTimeNano)

    def run(self):
        self.scSim.ExecuteSimulation()

    def _get_processed_data(self):
        """Retrieve and post-process log data (remove warmup, shift time)."""
        timeAxis = self.attErrorLog.times() * macros.NANO2SEC
        dataSigmaBR = self.attErrorLog.sigma_BR
        dataOmegaBR = self.attErrorLog.omega_BR_B
        dataTorque = self.torqueLog.torqueRequestBody
        
        warmup_time = self.sim_cfg.warmup_time
        mask = timeAxis >= warmup_time
        
        # Processed dictionary
        data = {
            "time": timeAxis[mask] - warmup_time,
            "sigma_BR": dataSigmaBR[mask],
            "omega_BR": dataOmegaBR[mask],
            "u_cmd": dataTorque[mask]
        }
        
        if self.sim_cfg.sc.actuators.use_rw:
            data["u_rw"] = self.rwMotorTorqueLog.motorTorque[mask]
            data["omega_rw"] = self.rwStateLog.wheelSpeeds[mask]
            
        return data

    def plot_results(self, save_dir=None):
        data = self._get_processed_data()
        
        if save_dir:
            fig_dir = os.path.join(save_dir, "figure")
            os.makedirs(fig_dir, exist_ok=True)
        else:
            fig_dir = None

        plot_attitude_error(data["time"], data["sigma_BR"])
        if fig_dir:
            plt.figure(1)
            plt.savefig(os.path.join(fig_dir, "attitude_error.png"))

        plot_rate_error(data["time"], data["omega_BR"])
        if fig_dir:
            plt.figure(2)
            plt.savefig(os.path.join(fig_dir, "rate_error.png"))

        plot_torque(data["time"], data["u_cmd"])
        if fig_dir:
            plt.figure(3)
            plt.savefig(os.path.join(fig_dir, "control_torque.png"))
        
        if self.sim_cfg.sc.actuators.use_rw:
            plot_rw_motor_torque(data["time"], data["u_rw"])
            if fig_dir:
                plt.figure(4)
                plt.savefig(os.path.join(fig_dir, "rw_motor_torque.png"))

            plot_rw_speeds(data["time"], data["omega_rw"])
            if fig_dir:
                plt.figure(5)
                plt.savefig(os.path.join(fig_dir, "rw_speeds.png"))
            
        # plt.show()

    def save_data(self, save_dir):
        """Save simulation data to .mat file."""
        data = self._get_processed_data()
        
        data_dir = os.path.join(save_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        mat_path = os.path.join(data_dir, "simulation_results.mat")
        sio.savemat(mat_path, data)
        print(f"[Sim] Data saved to {mat_path}")
