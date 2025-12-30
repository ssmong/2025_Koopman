import hydra
import numpy as np
from Basilisk.fswAlgorithms import (
    attTrackingError,
    inertial3D,
)
from Basilisk.simulation import simpleNav
from Basilisk.architecture import messaging
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import RigidBodyKinematics as rbk

class FSW:
    def __init__(self, bsk_sim):
        self.bsk_sim = bsk_sim
        self.sim_cfg = bsk_sim.sim_cfg
        self.ctl_cfg = bsk_sim.ctl_cfg
        
        self._setup_nav()
        self._setup_guidance()
        self._setup_control()
        self._connect_actuators()

    def _setup_nav(self):
        self.sNavObject = simpleNav.SimpleNav()
        self.sNavObject.ModelTag = "SimpleNavigation"
        # Priority 100
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.fswTaskName, self.sNavObject, 100)
        
        self.sNavObject.scStateInMsg.subscribeTo(self.bsk_sim.scObject.scStateOutMsg)

        noiseCfg = self.sim_cfg.sensors.nav_noise
        posErrorSigma = float(noiseCfg.pos)
        velErrorSigma = float(noiseCfg.vel)
        sigmaErrorSigma = float(noiseCfg.att)
        omegaErrorSigma = float(noiseCfg.rate)

        useNavError = (posErrorSigma > 0.0) or (velErrorSigma > 0.0) or (sigmaErrorSigma > 0.0) or (omegaErrorSigma > 0.0)
        if useNavError:
            PMatrix = np.zeros((18, 18))
            errorBounds = np.zeros(18)

            idx_pos = slice(0, 3)
            idx_vel = slice(3, 6)
            idx_sigma = slice(6, 9)
            idx_omega = slice(9, 12)

            if posErrorSigma > 0.0:
                PMatrix[range(0, 3), range(0, 3)] = posErrorSigma
                errorBounds[idx_pos] = 3.0 * posErrorSigma

            if velErrorSigma > 0.0:
                PMatrix[range(3, 6), range(3, 6)] = velErrorSigma
                errorBounds[idx_vel] = 3.0 * velErrorSigma

            if sigmaErrorSigma > 0.0:
                PMatrix[range(6, 9), range(6, 9)] = sigmaErrorSigma
                errorBounds[idx_sigma] = 3.0 * sigmaErrorSigma

            if omegaErrorSigma > 0.0:
                PMatrix[range(9, 12), range(9, 12)] = omegaErrorSigma
                errorBounds[idx_omega] = 3.0 * omegaErrorSigma

            self.sNavObject.PMatrix = PMatrix
            self.sNavObject.walkBounds = errorBounds

            rngSeed = getattr(self.sim_cfg.sensors, "rng_seed", None)
            if rngSeed is None:
                raise ValueError("sensors.rng_seed must be set when nav_noise is enabled (any sigma > 0).")
            self.sNavObject.RNGSeed = int(rngSeed)
        
        self.bsk_sim.sNavObject = self.sNavObject

    def _setup_guidance(self):
        self.inertial3DObj = inertial3D.inertial3D()
        self.inertial3DObj.ModelTag = "inertial3D"
        # Priority 90
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.fswTaskName, self.inertial3DObj, 90)
        
        # Convert Quaternion (Config) to MRP (Basilisk)
        q_R0N = list(self.sim_cfg.target_attitude)
        self.inertial3DObj.sigma_R0N = rbk.EP2MRP(q_R0N)

        self.attError = attTrackingError.attTrackingError()
        self.attError.ModelTag = "attErrorInertial3D"
        # Priority 80
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.fswTaskName, self.attError, 80)
        
        self.attError.attNavInMsg.subscribeTo(self.sNavObject.attOutMsg)
        self.attError.attRefInMsg.subscribeTo(self.inertial3DObj.attRefOutMsg)
        
        self.bsk_sim.attError = self.attError

    def _setup_control(self):
        self.controller = hydra.utils.instantiate(self.ctl_cfg)
        
        # Set ModelTag only if not already set by controller
        if not hasattr(self.controller, "ModelTag"):
            self.controller.ModelTag = "Controller"
        
        if hasattr(self.controller, 'set_warmup_time'):
            self.controller.set_warmup_time(self.sim_cfg.warmup_time)
        elif hasattr(self.controller, 'warmup_time'):
             self.controller.warmup_time = self.sim_cfg.warmup_time
             
        # Priority 70
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.fswTaskName, self.controller, 70)
        
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        vehicleConfigOut.ISCPntB_B = list(self.bsk_sim.sim_cfg.sc.inertia)
        self.vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
        
        if hasattr(self.controller, 'vehConfigInMsg'):
            self.controller.vehConfigInMsg.subscribeTo(self.vcMsg)
        
        if hasattr(self.controller, 'guidInMsg'):
            self.controller.guidInMsg.subscribeTo(self.attError.attGuidOutMsg)
            
        self.bsk_sim.controller = self.controller

    def _connect_actuators(self):
        use_rw = self.sim_cfg.sc.actuators.use_rw
        
        if use_rw:
            fswRwParamMsg = self.bsk_sim.dyn.rwFactory.getConfigMessage()
            
            self.bsk_sim.dyn.rwMotorTorqueObj.vehControlInMsg.subscribeTo(self.controller.cmdTorqueOutMsg)
            self.bsk_sim.dyn.rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
            self.bsk_sim.dyn.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.bsk_sim.dyn.rwMotorTorqueObj.rwMotorTorqueOutMsg)
            
            if hasattr(self.controller, 'rwParamsInMsg'):
                self.controller.rwParamsInMsg.subscribeTo(fswRwParamMsg)
            if hasattr(self.controller, 'rwSpeedsInMsg'):
                self.controller.rwSpeedsInMsg.subscribeTo(self.bsk_sim.dyn.rwStateEffector.rwSpeedOutMsg)

        else:
            self.bsk_sim.dyn.ctrlFTObject.cmdTorqueInMsg.subscribeTo(self.controller.cmdTorqueOutMsg)
