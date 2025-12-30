import os
import sys
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Basilisk imports
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.architecture import bskLogging # Import for logging
from Basilisk.architecture.bskLogging import BasiliskError

from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.fswAlgorithms import rwMotorTorque

from Basilisk.simulation import spacecraft
from Basilisk.simulation import exponentialAtmosphere
from Basilisk.simulation import facetDragDynamicEffector
from Basilisk.simulation import facetSRPDynamicEffector
from Basilisk.simulation import GravityGradientEffector
from Basilisk.simulation import linearSpringMassDamper
from Basilisk.simulation import fuelTank
from Basilisk.simulation import reactionWheelStateEffector
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import simpleNav
from Basilisk.simulation import simSynch

from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import simSetPlanetEnvironment
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import simIncludeRW
from Basilisk.utilities import vizSupport

# Custom Plotting Imports
from sim.utils.plot import (
    plot_attitude_error,
    plot_rate_error,
    plot_torque,
    plot_rw_motor_torque,
    plot_rw_speeds
)
from sim.utils.bks_utils import normalToDcmF0B

# Hydra Config Path
CONFIG_PATH = "../config" 
CONFIG_NAME = "evaluate"

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def run(cfg: DictConfig):
    # -------------------------------------------------------------------------
    # 0. Logger Setup
    # -------------------------------------------------------------------------
    # Use Basilisk Logger
    bsk_logger = bskLogging.BSKLogger()
    bsk_logger.setLogLevel(bskLogging.BSK_INFORMATION)
    bsk_logger.bskLog(bskLogging.BSK_INFORMATION, "Initializing Simulation in Single-Script Mode...")
    
    # -------------------------------------------------------------------------
    # 1. Configuration Extraction (Hydra -> Local Variables)
    # -------------------------------------------------------------------------
    # Simulation Parameters
    simulationTime = macros.sec2nano(cfg.sim.sim_time)
    simulationTimeStep = macros.sec2nano(cfg.sim.sim_dt)
    log_dt = macros.sec2nano(cfg.sim.log_dt)
    warmup_time = cfg.sim.warmup_time
    
    # Spacecraft Parameters
    sc_mass = cfg.sim.sc.mass
    sc_inertia = list(cfg.sim.sc.inertia)
    
    # Actuators
    useRW = cfg.sim.sc.actuators.use_rw
    
    # Sensors (Nav Noise)
    noiseCfg = cfg.sim.sensors.nav_noise
    useNavError = (float(noiseCfg.pos) > 0 or float(noiseCfg.vel) > 0 or 
                   float(noiseCfg.att) > 0 or float(noiseCfg.rate) > 0)
    
    show_plots = cfg.sim.show_plots
    
    use2StarTracker = False
    starTrackerFov = 20
    sunSensorFov = 50

    # -------------------------------------------------------------------------
    # 2. Experiment Setup (Hydra Output Dir)
    # -------------------------------------------------------------------------
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except Exception as e:
        bsk_logger.bskLog(bskLogging.BSK_WARNING, f"Failed to get Hydra output directory: {e}. Using current directory.")
        output_dir = "."
    
    figureDir = os.path.join(output_dir, "figure")
    dataDir = os.path.join(output_dir, "data")
    os.makedirs(figureDir, exist_ok=True)
    os.makedirs(dataDir, exist_ok=True)

    rngSeed = 42
    simTaskName = "simTask"
    simProcessName = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(False)

    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # -------------------------------------------------------------------------
    # 3. Spacecraft Setup
    # -------------------------------------------------------------------------
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = cfg.sim.sc.tag
    
    scObject.hub.mHub = sc_mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(sc_inertia)

    # Orbital Elements (From Config)
    orbit_cfg = cfg.sim.sc.orbit
    oe = orbitalMotion.ClassicElements()
    oe.a = (orbit_cfg.R_E + orbit_cfg.a) * 1000  # km -> m
    oe.e = orbit_cfg.e
    oe.i = orbit_cfg.i * macros.D2R
    oe.Omega = orbit_cfg.Omega * macros.D2R
    oe.omega = orbit_cfg.omega * macros.D2R
    oe.f = orbit_cfg.f * macros.D2R

    # Initial Attitude (From Config)
    att_cfg = cfg.sim.sc.init_attitude
    q_BNInit = list(att_cfg.q_BN)
    scObject.hub.sigma_BNInit = rbk.EP2MRP(q_BNInit)
    scObject.hub.omega_BN_BInit = list(att_cfg.omega_BN_B)

    scSim.AddModelToTask(simTaskName, scObject, 1)

    # -------------------------------------------------------------------------
    # 4. Environment Setup
    # -------------------------------------------------------------------------
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies('earth', 'sun')
    earth = gravBodies['earth']
    earth.useSphericalHarmonicsGravityModel(bskPath + '/supportData/LocalGravData/GGM03S.txt', 4)
    
    mu = earth.mu
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    
    gravFactory.addBodiesTo(scObject)

    timeInitString = '2025 DECEMBER 09 00:00:00.0'
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(simTaskName, gravFactory.spiceObject, 2)
    
    sunIdx = 1

    atmo = exponentialAtmosphere.ExponentialAtmosphere()
    simSetPlanetEnvironment.exponentialAtmosphere(atmo, 'earth')
    scSim.AddModelToTask(simTaskName, atmo, 2)
    atmo.addSpacecraftToModel(scObject.scStateOutMsg)


    # -------------------------------------------------------------------------
    # 5. Disturbance Setup
    # -------------------------------------------------------------------------
    # Geometry Config
    geom_cfg = cfg.sim.sc.geometry
    hub_cfg = geom_cfg.hub
    panel_cfg = geom_cfg.panel

    # 5.1 Drag
    dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
    dragEffector.ModelTag = "FacetDrag"
    dragEffector.atmoDensInMsg.subscribeTo(atmo.envOutMsgs[0])

    # Hub Parameters
    hubDragCoeff = hub_cfg.coeff
    hubSize = hub_cfg.size
    hubArea = hub_cfg.area
    hubOffSet = hubSize / 2.0
    hubNormals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    
    for normal in hubNormals:
        location = normal * hubOffSet
        dragEffector.addFacet(hubArea, hubDragCoeff, normal, location)
        
    # Panel Parameters
    panelArea = panel_cfg.area
    panelCoeff = panel_cfg.coeff
    panelDist = panel_cfg.offset
    
    panel_data = [
        (np.array([0, 0,  1]), np.array([0,  panelDist, 0])),
        (np.array([0, 0, -1]), np.array([0,  panelDist, 0])),
        (np.array([0, 0,  1]), np.array([0, -panelDist, 0])),
        (np.array([0, 0, -1]), np.array([0, -panelDist, 0]))
    ]
    for normal, loc in panel_data:
        dragEffector.addFacet(panelArea, panelCoeff, normal, loc)
    
    scObject.addDynamicEffector(dragEffector)
    scSim.AddModelToTask(simTaskName, dragEffector, 2)

    # 5.2 SRP
    srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
    srpEffector.ModelTag = "FacetSRP"
    srpEffector.setNumFacets(10)
    srpEffector.setNumArticulatedFacets(0)
    srpEffector.sunInMsg.subscribeTo(gravFactory.spiceObject.planetStateOutMsgs[sunIdx])
    
    # Hub SRP
    hubDiffuseCoeff = hub_cfg.diff
    hubSpecularCoeff = hub_cfg.spec
    for normal in hubNormals:
        location = normal * hubOffSet
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])
        srpEffector.addFacet(hubArea, dcm_F0B, nHat_F, rotHat_F, location, hubDiffuseCoeff, hubSpecularCoeff)

    # Panel SRP
    panelDiffuseCoeff = panel_cfg.diff
    panelSpecularCoeff = panel_cfg.spec
    for normal, loc in panel_data:
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])
        srpEffector.addFacet(panelArea, dcm_F0B, nHat_F, rotHat_F, loc, panelDiffuseCoeff, panelSpecularCoeff)
        
    scObject.addDynamicEffector(srpEffector)
    scSim.AddModelToTask(simTaskName, srpEffector, 2)

    # 5.3 Gravity Gradient
    ggEffector = GravityGradientEffector.GravityGradientEffector()
    ggEffector.ModelTag = "GravityGradient"
    ggEffector.addPlanetName(earth.planetName)
    scObject.addDynamicEffector(ggEffector)
    scSim.AddModelToTask(simTaskName, ggEffector)

    # 5.4 Sloshing
    slosh_cfg = cfg.sim.sloshing
    target_class = slosh_cfg.get("_target_", "sim.setup.sloshing.NoSloshing")
    
    tank = fuelTank.FuelTank()
    tankModel = fuelTank.FuelTankModelConstantVolume()
    tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
    tankModel.radiusTankInit = 0.5
    
    particles = []
    
    if "SpringMassSloshing" in target_class:
        bsk_logger.bskLog(bskLogging.BSK_INFORMATION, f"Sloshing Model: SpringMassSloshing")
        
        propMassInit = slosh_cfg.mass_init
        r_TB_B = slosh_cfg.r_TB_B
        particle_list = slosh_cfg.particles
        
        for p_cfg in particle_list:
            particle = linearSpringMassDamper.LinearSpringMassDamper()
            particle.k = p_cfg.k
            particle.c = p_cfg.c
            particle.massInit = p_cfg.mass
            particle.rhoInit = p_cfg.get("rho", 0.0)
            particle.rhoDotInit = p_cfg.get("rho_dot", 0.0)
            
            pos = p_cfg.pos
            direction = p_cfg.dir
            particle.r_PB_B = [[pos[0]], [pos[1]], [pos[2]]]
            particle.pHat_B = [[direction[0]], [direction[1]], [direction[2]]]
            
            particles.append(particle)
            scObject.addStateEffector(particle)
            
        tankModel.propMassInit = propMassInit
        tank.r_TB_B = [[r_TB_B[0]], [r_TB_B[1]], [r_TB_B[2]]]
        
    elif "NoSloshing" in target_class:
        bsk_logger.bskLog(bskLogging.BSK_INFORMATION, "Sloshing Model: NoSloshing")
        tankModel.propMassInit = slosh_cfg.get("mass_init", 100.0)
        r_TB_B = slosh_cfg.get("r_TB_B", [0.0, 0.0, 0.1])
        tank.r_TB_B = [[r_TB_B[0]], [r_TB_B[1]], [r_TB_B[2]]]
        
    elif "MPBMSloshing" in target_class:
        raise NotImplementedError("MPBM Sloshing is not implemented yet in run_sim_temp.py")
        
    else:
        bsk_logger.bskLog(bskLogging.BSK_WARNING, f"Unknown Sloshing target: {target_class}. Defaulting to NoSloshing.")
        tankModel.propMassInit = 100.0
        tank.r_TB_B = [[0.0], [0.0], [0.1]]

    tank.setTankModel(tankModel)
    tank.nameOfMassState = "fuelTankMass"
    tank.updateOnly = True
    
    for particle in particles:
        tank.pushFuelSloshParticle(particle)
        
    scObject.addStateEffector(tank)

    # -------------------------------------------------------------------------
    # 6. Actuators
    # -------------------------------------------------------------------------
    if useRW:
        bsk_logger.bskLog(bskLogging.BSK_INFORMATION, "Reaction Wheels Enabled.")
        rwFactory = simIncludeRW.rwFactory()
        varRWModel = messaging.BalancedWheels
        RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100., RWModel=varRWModel)
        RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200., RWModel=varRWModel)
        RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300., RWModel=varRWModel)
        
        numRW = rwFactory.getNumOfDevices()
        rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwStateEffector.ModelTag = "RW_cluster"
        rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
        scSim.AddModelToTask(simTaskName, rwStateEffector, 2)
        
        rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
        rwMotorTorqueObj.ModelTag = "RWMotorTorque"
        scSim.AddModelToTask(simTaskName, rwMotorTorqueObj)
        rwMotorTorqueObj.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    else:
        bsk_logger.bskLog(bskLogging.BSK_INFORMATION, "External Force/Torque Enabled.")
        ctrlFTObject = extForceTorque.ExtForceTorque()
        ctrlFTObject.ModelTag = "controlForceTorque"
        scObject.addDynamicEffector(ctrlFTObject)
        scSim.AddModelToTask(simTaskName, ctrlFTObject)

    # -------------------------------------------------------------------------
    # 7. Navigation
    # -------------------------------------------------------------------------
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)
    
    if useNavError:
        bsk_logger.bskLog(bskLogging.BSK_INFORMATION, "Navigation Noise Enabled.")
        posError = float(noiseCfg.pos)
        velError = float(noiseCfg.vel)
        attErrorSigma = float(noiseCfg.att) * macros.D2R
        rateError = float(noiseCfg.rate) * macros.D2R
        
        PMatrix = np.zeros((18, 18))
        errorBounds = np.zeros(18)
        
        if posError > 0:
            PMatrix[range(0,3), range(0,3)] = posError
            errorBounds[0:3] = 3 * posError
        if velError > 0:
            PMatrix[range(3,6), range(3,6)] = velError
            errorBounds[3:6] = 3 * velError
        if attErrorSigma > 0:
            PMatrix[range(6,9), range(6,9)] = attErrorSigma
            errorBounds[6:9] = 3 * attErrorSigma
        if rateError > 0:
            PMatrix[range(9,12), range(9,12)] = rateError
            errorBounds[9:12] = 3 * rateError
            
        sNavObject.PMatrix = PMatrix
        sNavObject.walkBounds = errorBounds
        sNavObject.RNGSeed = rngSeed

    # -------------------------------------------------------------------------
    # 8. FSW & Controller
    # -------------------------------------------------------------------------
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DObj)
    inertial3DObj.sigma_R0N = [0., 0., 0.]

    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)

    # Controller Selection Logic using _target_
    ctl_target = cfg.controller.get("_target_", "RandomTorque")
    
    if "BskKoopmanMPC" in ctl_target:
        from sim.controller.koopman_mpc import BskKoopmanMPC
        
        mpc_params = dict(cfg.controller.mpc_params)
            
        torqueControl = BskKoopmanMPC(
            checkpoint_dir=cfg.controller.checkpoint_dir,
            mpc_params=mpc_params,
            device=cfg.controller.get("device", "cpu")
        )
        torqueControl.ModelTag = "KoopmanMPC"
        scSim.AddModelToTask(simTaskName, torqueControl)
        
        if hasattr(torqueControl, 'set_warmup_time'):
            torqueControl.set_warmup_time(warmup_time)
            
    else:
        # Default to Random
        bsk_logger.bskLog(bskLogging.BSK_INFORMATION, f"Controller type (Target: {ctl_target}) is unknown. Defaulting to Random Torque Controller")
        from Basilisk.ExternalModules import randomTorque
        torqueControl = randomTorque.RandomTorque()
        torqueControl.ModelTag = "randomTorque"
        scSim.AddModelToTask(simTaskName, torqueControl)
        torqueControl.setTorqueMagnitude(1)
        torqueControl.setSeed(rngSeed)

    # -------------------------------------------------------------------------
    # 9. Message Linking & Logging
    # -------------------------------------------------------------------------
    samplingTime = log_dt
    
    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    torqueLog = torqueControl.cmdTorqueOutMsg.recorder(samplingTime)
    sNavRec = sNavObject.attOutMsg.recorder(samplingTime)
    dataRec = scObject.scStateOutMsg.recorder(samplingTime)
    
    scSim.AddModelToTask(simTaskName, attErrorLog)
    scSim.AddModelToTask(simTaskName, torqueLog)
    scSim.AddModelToTask(simTaskName, sNavRec)
    scSim.AddModelToTask(simTaskName, dataRec)
    
    if useRW:
        rwMotorTorqueLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
        rwStateLog = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
        scSim.AddModelToTask(simTaskName, rwMotorTorqueLog)
        scSim.AddModelToTask(simTaskName, rwStateLog)
        
        rwLogs = []
        for item in range(numRW):
            rwLogs.append(rwStateEffector.rwOutMsgs[item].recorder(samplingTime))
            scSim.AddModelToTask(simTaskName, rwLogs[item])

    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    vehicleConfigOut.ISCPntB_B = sc_inertia
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
    torqueControl.vehConfigInMsg.subscribeTo(vcMsg)
    torqueControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    
    if useRW:
        fswRwParamMsg = rwFactory.getConfigMessage()
        
        torqueControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
        torqueControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
        rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
        rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
        rwMotorTorqueObj.vehControlInMsg.subscribeTo(torqueControl.cmdTorqueOutMsg)
    else:
        ctrlFTObject.cmdTorqueInMsg.subscribeTo(torqueControl.cmdTorqueOutMsg)

    # -------------------------------------------------------------------------
    # 10. Visualization (Vizard)
    # -------------------------------------------------------------------------
    useVizard = cfg.sim.use_vizard and vizSupport.vizFound
    if useVizard:
        clockSync = simSynch.ClockSynch()
        clockSync.accelFactor = 5.0
        scSim.AddModelToTask(simTaskName, clockSync)
        
        viz = vizSupport.enableUnityVisualization(
            scSim, simTaskName, scObject,
            rwEffectorList=[rwStateEffector] if useRW else None, # List Wrapper
            liveStream=True,
            broadcastStream=False,
            )
        viz.settings.showRWLabels = 0  
        viz.settings.viewRWHUD = 1
        
        vizSupport.createConeInOut(viz, toBodyName='sun', coneColor = 'r',
                           normalVector_B=[1, 0, 0], incidenceAngle=starTrackerFov*macros.D2R, isKeepIn=False,
                           coneHeight=5.0, coneName='sunKeepOut')
        if use2StarTracker:
            vizSupport.createConeInOut(viz, toBodyName='sun', coneColor = 'm',
                               normalVector_B=[0, 0, 1], incidenceAngle=starTrackerFov*macros.D2R, isKeepIn=True,
                               coneHeight=5.0, coneName='sunKeepOut')
        vizSupport.createConeInOut(viz, toBodyName='sun', coneColor = 'b',
                           normalVector_B=[0, 1, 0], incidenceAngle=sunSensorFov*macros.D2R, isKeepIn=True,
                           coneHeight=5.0, coneName='sunKeepIn')

    # -------------------------------------------------------------------------
    # 11. Execution
    # -------------------------------------------------------------------------
    bsk_logger.bskLog(bskLogging.BSK_INFORMATION, "Starting Simulation...")
    scSim.InitializeSimulation()
    
    if useVizard:
        currentSimNanos = 0
        while currentSimNanos < simulationTime:
            if vizSupport.vizFound and vizSupport.endFlag:
                break
            currentSimNanos += simulationTimeStep
            scSim.ConfigureStopTime(currentSimNanos)
            scSim.ExecuteSimulation()
    else:
        scSim.ConfigureStopTime(simulationTime)
        scSim.ExecuteSimulation()
    
    bsk_logger.bskLog(bskLogging.BSK_INFORMATION, "Simulation Finished.")

    # -------------------------------------------------------------------------
    # 12. Profiling Stats
    # -------------------------------------------------------------------------
    if hasattr(torqueControl, "profiler"):
        controller_name = getattr(torqueControl, "ModelTag", "Controller")
        torqueControl.profiler.print_stats(name=controller_name)

    # -------------------------------------------------------------------------
    # 13. Data Processing & Saving
    # -------------------------------------------------------------------------
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
    dataTorque = torqueLog.torqueRequestBody
    timeAxis = attErrorLog.times() * macros.NANO2SEC
    
    # Filter warmup
    mask = timeAxis >= warmup_time
    
    # Save processed data
    save_data = {
        "time": timeAxis[mask] - warmup_time,
        "sigma_BR": dataSigmaBR[mask],
        "omega_BR": dataOmegaBR[mask],
        "u_cmd": dataTorque[mask]
    }
    
    if useRW:
        save_data["u_rw"] = rwMotorTorqueLog.motorTorque[mask]
        save_data["omega_rw"] = rwStateLog.wheelSpeeds[mask]
        
    mat_path = os.path.join(dataDir, "simulation_results.mat")
    sio.savemat(mat_path, save_data)
    bsk_logger.bskLog(bskLogging.BSK_INFORMATION, f"Data saved to {mat_path}")

    # -------------------------------------------------------------------------
    # 14. Plotting
    # -------------------------------------------------------------------------
    # Plot 1: Attitude Error
    plot_attitude_error(save_data["time"], save_data["sigma_BR"], position=(100, 100))
    plt.savefig(os.path.join(figureDir, "attitude_error.png"), dpi=150, bbox_inches='tight')
    
    # Plot 2: Rate Error
    plot_rate_error(save_data["time"], save_data["omega_BR"], position=(100, 800))
    plt.savefig(os.path.join(figureDir, "rate_error.png"), dpi=150, bbox_inches='tight')
    
    # Plot 3: Control Torque
    plot_torque(save_data["time"], save_data["u_cmd"], position=(1100, 100))
    plt.savefig(os.path.join(figureDir, "torque.png"), dpi=150, bbox_inches='tight')
    
    if useRW:
        # Plot 4: RW Motor Torque
        plot_rw_motor_torque(save_data["time"], save_data["u_rw"], position=(100, 500))
        plt.savefig(os.path.join(figureDir, "rw_motor_torque.png"), dpi=150, bbox_inches='tight')
        
        # Plot 5: RW Speed
        plot_rw_speeds(save_data["time"], save_data["omega_rw"], position=(1100, 500))
        plt.savefig(os.path.join(figureDir, "rw_speed.png"), dpi=150, bbox_inches='tight')

    if show_plots:
        plt.show()
    
    plt.close('all')

if __name__ == "__main__":
    run()
