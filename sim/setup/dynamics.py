import hydra
from Basilisk.simulation import (
    spacecraft,
    reactionWheelStateEffector,
    extForceTorque,
    rwMotorTorque,
)
from Basilisk.utilities import (
    unitTestSupport,
    RigidBodyKinematics as rbk,
    simIncludeRW,
    macros,
)
from Basilisk.architecture import messaging

class Dynamics:
    def __init__(self, bsk_sim):
        self.bsk_sim = bsk_sim
        self.sim_cfg = bsk_sim.sim_cfg
        self.sc_cfg = self.sim_cfg.sc
        
        self._setup_sc()
        self._setup_sloshing()
        self._setup_actuators()

    def _setup_sc(self):
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = self.sc_cfg.tag

        self.scObject.hub.mHub = self.sc_cfg.mass
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(self.sc_cfg.inertia)

        q_BN = self.sc_cfg.init_attitude.q_BN
        sigma_BN = rbk.EP2MRP(q_BN)
        self.scObject.hub.sigma_BNInit = sigma_BN
        self.scObject.hub.omega_BN_BInit = self.sc_cfg.init_attitude.omega_BN_B
        
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, self.scObject, 1)
        
        self.bsk_sim.scObject = self.scObject
        
    def _setup_sloshing(self):
        self.sloshing_model = hydra.utils.instantiate(self.sim_cfg.sloshing)
        self.tank = self.sloshing_model.setup(self.scObject)

    def _setup_actuators(self):
        self.use_rw = self.sc_cfg.actuators.use_rw
        
        if self.use_rw:
            rwFactory = simIncludeRW.rwFactory()
            varRWModel = messaging.BalancedWheels
            
            RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100., RWModel=varRWModel)
            RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200., RWModel=varRWModel)
            RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300., RWModel=varRWModel)
            
            self.numRW = rwFactory.getNumOfDevices()
            self.rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
            self.rwStateEffector.ModelTag = "RW_cluster"
            rwFactory.addToSpacecraft(self.scObject.ModelTag, self.rwStateEffector, self.scObject)
            self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, self.rwStateEffector, 2)
            
            self.rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
            self.rwMotorTorqueObj.ModelTag = "RWMotorTorque"
            self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, self.rwMotorTorqueObj)
            
            self.rwMotorTorqueObj.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            self.rwFactory = rwFactory
            
        else:
            self.ctrlFTObject = extForceTorque.ExtForceTorque()
            self.ctrlFTObject.ModelTag = "controlForceTorque"
            self.scObject.addDynamicEffector(self.ctrlFTObject)
            self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, self.ctrlFTObject)
