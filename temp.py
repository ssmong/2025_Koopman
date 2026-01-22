import os
import numpy as np
import h5py
import random
import shutil
import argparse
from datetime import datetime

# Basilisk imports
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.architecture import messaging
from Basilisk.architecture import bskLogging

# Simulation modules
from Basilisk.simulation import spacecraft
from Basilisk.simulation import exponentialAtmosphere
from Basilisk.simulation import facetDragDynamicEffector
from Basilisk.simulation import facetSRPDynamicEffector
from Basilisk.simulation import GravityGradientEffector
from Basilisk.simulation import simpleNav
from Basilisk.simulation import extForceTorque
from Basilisk.simulation import fuelTank
from Basilisk.simulation import linearSpringMassDamper

# FSW modules
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import inertial3D

# External modules
from Basilisk.ExternalModules import randomTorque
from Basilisk.ExternalModules import movingPulsatingBall

from Basilisk import __path__

# Monte Carlo imports
from Basilisk.utilities.MonteCarlo.Controller import Controller, RetentionPolicy

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# Module-level global variables
MC_CTRL_DT = 1.0
MC_LOG_DT = 0.1
MC_SIM_DT = 0.01
MC_SIM_TIME = 1000.0
MC_SLOSHING_MODEL = "none"
MC_VALIDATION = False

def normalToDcmF0B(nHat_B):
    """B frame 법선 벡터로부터 dcm_F0B 계산 (F frame +Y = nHat_B)"""
    nHat_B = np.array(nHat_B) / np.linalg.norm(nHat_B)
    if np.allclose(nHat_B, [0, 1, 0]):
        return np.eye(3)
    elif np.allclose(nHat_B, [0, -1, 0]):
        return rbk.PRV2C(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        y_axis = nHat_B
        temp = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x_axis = temp - np.dot(temp, y_axis) * y_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis) / np.linalg.norm(np.cross(x_axis, y_axis))
        return np.column_stack([x_axis, y_axis, z_axis]).T

def createScenario():
    global MC_CTRL_DT, MC_LOG_DT, MC_SIM_DT, MC_SIM_TIME, MC_SLOSHING_MODEL
    ctrlDtNano = macros.sec2nano(MC_CTRL_DT)
    logDtNano = macros.sec2nano(MC_LOG_DT)
    simDtNano = macros.sec2nano(MC_SIM_DT)
    simTimeNano = macros.sec2nano(MC_SIM_TIME)
    
    scSim = SimulationBaseClass.SimBaseClass()
    
    navTaskName = "navTask"
    dynTaskName = "dynTask"
    ctrlTaskName = "ctrlTask"
    logTaskName = "logTask"
    simProcessName = "simProcess"
    
    scSim.dynProcess = scSim.CreateNewProcess(simProcessName)
    
    # Task Scheduling
    scSim.dynProcess.addTask(scSim.CreateNewTask(navTaskName, logDtNano))
    scSim.dynProcess.addTask(scSim.CreateNewTask(ctrlTaskName, logDtNano))
    scSim.dynProcess.addTask(scSim.CreateNewTask(logTaskName, logDtNano))
    scSim.dynProcess.addTask(scSim.CreateNewTask(dynTaskName, simDtNano))

    # --- 1. Spacecraft ---
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "SMRL-Sat"
    I = [101.67, 0., 0., 0., 135.42, 0., 0., 0., 153.75]
    scObject.hub.mHub = 500.0
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    
    oe = orbitalMotion.ClassicElements()
    oe.a = (6378.1366 + 402.72) * 1000
    oe.e = 0.00130547
    oe.i = 51.60 * macros.D2R
    oe.Omega = 198.38 * macros.D2R
    oe.omega = 39.26 * macros.D2R
    oe.f = 117.71 * macros.D2R
    
    u = np.random.rand(3)
    q_rand = np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
    ])
    if q_rand[0] < 0: q_rand = -q_rand
    
    scObject.hub.sigma_BNInit = rbk.EP2MRP(q_rand)
    scObject.hub.omega_BN_BInit = np.random.uniform(-0.2, 0.2, size=(3, 1)).tolist()

    scSim.AddModelToTask(dynTaskName, scObject, 1)
    scSim.scObject = scObject

    # --- 2. Environment ---
    scSim.gravFactory = simIncludeGravBody.gravBodyFactory()
    scSim.gravBodies = scSim.gravFactory.createBodies('earth', 'sun')
    scSim.earth = scSim.gravBodies['earth']
    scSim.earth.useSphericalHarmonicsGravityModel(bskPath + '/supportData/LocalGravData/GGM03S.txt', 4)
    
    mu = scSim.earth.mu
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    scSim.gravFactory.addBodiesTo(scObject)
    
    timeInitString = '2025 DECEMBER 09 00:00:00.0'
    scSim.spiceObject = scSim.gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    scSim.spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(dynTaskName, scSim.gravFactory.spiceObject, 2)
    
    scSim.atmo = exponentialAtmosphere.ExponentialAtmosphere()
    scSim.atmo.ModelTag = "exponentialAtmosphere"
    scSim.atmo.planetRadius = 6378136.6
    scSim.atmo.scaleHeight = 7200.0
    scSim.atmo.baseDensity = 1.217
    scSim.atmo.addSpacecraftToModel(scObject.scStateOutMsg)
    scSim.AddModelToTask(dynTaskName, scSim.atmo)

    # --- 3. Disturbances ---
    scSim.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
    hubDragCoeff = 2.2
    hubSize = 1.0
    hubArea = hubSize ** 2.0
    hubOffSet = hubSize / 2.0
    hubNormals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    for normal in hubNormals:
        location = normal * hubOffSet
        scSim.dragEffector.addFacet(hubArea, hubDragCoeff, normal, location)
    
    panelArea, panelCoeff = 2.0, 2.2
    panelDist = hubOffSet + 1.0
    panel_data = [
        (np.array([0, 0,  1]), np.array([0,  panelDist, 0])),
        (np.array([0, 0, -1]), np.array([0,  panelDist, 0])),
        (np.array([0, 0,  1]), np.array([0, -panelDist, 0])),
        (np.array([0, 0, -1]), np.array([0, -panelDist, 0]))
    ]
    for normal, loc in panel_data:
        scSim.dragEffector.addFacet(panelArea, panelCoeff, normal, loc)
    
    scSim.dragEffector.atmoDensInMsg.subscribeTo(scSim.atmo.envOutMsgs[0])
    scObject.addDynamicEffector(scSim.dragEffector)
    scSim.AddModelToTask(dynTaskName, scSim.dragEffector)

    # SRP
    scSim.srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
    scSim.srpEffector.ModelTag = "FacetSRP"
    scSim.srpEffector.setNumFacets(10)
    scSim.srpEffector.sunInMsg.subscribeTo(scSim.gravFactory.spiceObject.planetStateOutMsgs[1])
    
    hubDiffuseCoeff = 0.1
    hubSpecularCoeff = 0.9
    for normal in hubNormals:
        location = normal * hubOffSet
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])
        scSim.srpEffector.addFacet(hubArea, dcm_F0B, nHat_F, rotHat_F, location, 
                                    hubDiffuseCoeff, hubSpecularCoeff)
                            
    panelDiffuseCoeff = 0.16
    panelSpecularCoeff = 0.16
    for normal, loc in panel_data:
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])
        scSim.srpEffector.addFacet(panelArea, dcm_F0B, nHat_F, rotHat_F, loc,
                                    panelDiffuseCoeff, panelSpecularCoeff)
        
    scObject.addDynamicEffector(scSim.srpEffector)
    scSim.AddModelToTask(dynTaskName, scSim.srpEffector)

    # Gravity Gradient
    scSim.ggEffector = GravityGradientEffector.GravityGradientEffector()
    scSim.ggEffector.ModelTag = "GravityGradient"
    scSim.ggEffector.addPlanetName(scSim.earth.planetName)
    scObject.addDynamicEffector(scSim.ggEffector)
    scSim.AddModelToTask(dynTaskName, scSim.ggEffector)

    # --- Sloshing ---
    sloshingModel = MC_SLOSHING_MODEL
    if sloshingModel == "mpbm":
        scSim.mpbm = movingPulsatingBall.MovingPulsatingBall()
        scSim.mpbm.ModelTag = "mpbm"
        scSim.mpbm.massInit = 100.0
        scSim.mpbm.radiusTank = 0.5
        # [Alignment] Tank Center = Body Center
        scSim.mpbm.r_TB_B = [[0.0], [0.0], [0.0]]
        
        # [Random Initialization]
        # Position: Inside tank (random direction, magnitude < 0.2m)
        r_dir = np.random.normal(0, 1, 3)
        r_dir /= np.linalg.norm(r_dir)
        r_mag = np.random.uniform(0.0, 0.2)
        r_init = r_dir * r_mag
        scSim.mpbm.r_Init_B = [[r_init[0]], [r_init[1]], [r_init[2]]]
        
        # Velocity: Random velocity within [-0.1, 0.1] m/s
        v_init = np.random.uniform(-0.1, 0.1, 3)
        scSim.mpbm.v_Init_B = [[v_init[0]], [v_init[1]], [v_init[2]]]

        # Omega: Random angular velocity within [-0.2, 0.2] rad/s
        # This ensures initial spin/tumble exists in all axes including Z
        omega_init = np.random.uniform(-0.2, 0.2, 3)
        scSim.mpbm.omega_Init_B = [[omega_init[0]], [omega_init[1]], [omega_init[2]]]

        scObject.addStateEffector(scSim.mpbm)
        scSim.AddModelToTask(dynTaskName, scSim.mpbm)
    elif sloshingModel == "spring":
        scSim.tank = fuelTank.FuelTank()
        scSim.tankModel = fuelTank.FuelTankModelConstantVolume()
        scSim.tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        scSim.tankModel.radiusTankInit = 0.5
        scSim.particles = []
        directions = [[1,0,0], [0,1,0], [0,0,1]]
        positions = [[0.1,0,-0.1], [0,0,0.1], [-0.1,0,0.1]]
        for i, (direction, position) in enumerate(zip(directions, positions)):
            particle = linearSpringMassDamper.LinearSpringMassDamper()
            particle.k = 0.2
            particle.c = 0.05
            particle.r_PB_B = [[position[0]], [position[1]], [position[2]]]
            particle.pHat_B = [[direction[0]], [direction[1]], [direction[2]]]
            particle.rhoInit = 0.05 if i == 0 else -0.025
            particle.rhoDotInit = 0.0
            particle.massInit = 10.0
            scSim.particles.append(particle)
        scSim.tankModel.propMassInit = 70.0
        scSim.tank.setTankModel(scSim.tankModel)
        scSim.tank.r_TB_B = [[0], [0], [0.1]]
        scSim.tank.nameOfMassState = "fuelTankMass"
        scSim.tank.updateOnly = True
        for particle in scSim.particles:
            scSim.tank.pushFuelSloshParticle(particle)
            scObject.addStateEffector(particle)
        scObject.addStateEffector(scSim.tank)
    else:
        scSim.tank = fuelTank.FuelTank()
        scSim.tankModel = fuelTank.FuelTankModelConstantVolume()
        scSim.tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        scSim.tankModel.radiusTankInit = 0.5
        scSim.tankModel.propMassInit = 100.0
        scSim.tank.setTankModel(scSim.tankModel)
        scSim.tank.r_TB_B = [[0], [0], [0.1]]
        scSim.tank.nameOfMassState = "fuelTankMass"
        scSim.tank.updateOnly = True
        scObject.addStateEffector(scSim.tank)

    # --- 4. Navigation & Control ---
    scSim.sNavObject = simpleNav.SimpleNav()
    scSim.sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(navTaskName, scSim.sNavObject)
    scSim.sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    scSim.inertial3DObj = inertial3D.inertial3D()
    scSim.inertial3DObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(navTaskName, scSim.inertial3DObj)
    scSim.inertial3DObj.sigma_R0N = [0., 0., 0.]

    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(navTaskName, attError)
    attError.attNavInMsg.subscribeTo(scSim.sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(scSim.inertial3DObj.attRefOutMsg)
    scSim.attError = attError

    # [Random Torque Control Setup]
    scSim.rngControl = randomTorque.RandomTorque()
    scSim.rngControl.ModelTag = "randomTorque"
    scSim.rngControl.setTorqueMagnitude(2)
    
    scSim.rngControl.setHoldPeriod(MC_CTRL_DT) 
    
    scSim.AddModelToTask(ctrlTaskName, scSim.rngControl)
    
    scSim.vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
    scSim.configDataMsg = messaging.VehicleConfigMsg().write(scSim.vehicleConfigOut)
    scSim.rngControl.vehConfigInMsg.subscribeTo(scSim.configDataMsg)
    scSim.rngControl.guidInMsg.subscribeTo(scSim.attError.attGuidOutMsg)
    
    scSim.ctrlFTObject = extForceTorque.ExtForceTorque()
    scSim.ctrlFTObject.ModelTag = "controlForceTorque"
    scObject.addDynamicEffector(scSim.ctrlFTObject)
    scSim.AddModelToTask(dynTaskName, scSim.ctrlFTObject)
    scSim.ctrlFTObject.cmdTorqueInMsg.subscribeTo(scSim.rngControl.cmdTorqueOutMsg)
    
    # --- Logging ---
    scSim.msgRecList = {}
    scSim.msgRecList["attError.attGuidOutMsg"] = scSim.attError.attGuidOutMsg.recorder(logDtNano)
    scSim.AddModelToTask(logTaskName, scSim.msgRecList["attError.attGuidOutMsg"])
    scSim.msgRecList["rngControl.cmdTorqueOutMsg"] = scSim.rngControl.cmdTorqueOutMsg.recorder(logDtNano)
    scSim.AddModelToTask(logTaskName, scSim.msgRecList["rngControl.cmdTorqueOutMsg"])

    if MC_VALIDATION and sloshingModel == "mpbm":
        scSim.msgRecList["mpbm.mpbmOutMsg"] = scSim.mpbm.mpbmOutMsg.recorder(logDtNano)
        scSim.AddModelToTask(logTaskName, scSim.msgRecList["mpbm.mpbmOutMsg"])
    
    scSim.simulationTime = simTimeNano
    scSim.samplingTime = logDtNano
    
    return scSim

def executeScenario(sim):
    sim.InitializeSimulation()
    sim.ConfigureStopTime(sim.simulationTime)
    sim.ExecuteSimulation()

def check_for_nans(data):
    if "messages" not in data:
        return False
    for key, val in data["messages"].items():
        if np.isnan(val).any():
            return True
    return False

def run_single_retry():
    sim = createScenario()
    executeScenario(sim)
    messages = {}
    
    rec_att = sim.msgRecList["attError.attGuidOutMsg"]
    times = rec_att.times() 
    sigma_BR = unitTestSupport.addTimeColumn(times, rec_att.sigma_BR)
    omega_BR_B = unitTestSupport.addTimeColumn(times, rec_att.omega_BR_B)
    messages["attError.attGuidOutMsg.sigma_BR"] = sigma_BR
    messages["attError.attGuidOutMsg.omega_BR_B"] = omega_BR_B
    
    rec_trq = sim.msgRecList["rngControl.cmdTorqueOutMsg"]
    torque = unitTestSupport.addTimeColumn(rec_trq.times(), rec_trq.torqueRequestBody)
    messages["rngControl.cmdTorqueOutMsg.torqueRequestBody"] = torque
    
    if MC_VALIDATION and MC_SLOSHING_MODEL == "mpbm":
        rec_mpbm = sim.msgRecList["mpbm.mpbmOutMsg"]
        r_slug = unitTestSupport.addTimeColumn(rec_mpbm.times(), rec_mpbm.r_Slug_B)
        v_slug = unitTestSupport.addTimeColumn(rec_mpbm.times(), rec_mpbm.v_Slug_B)
        torque_int = unitTestSupport.addTimeColumn(rec_mpbm.times(), rec_mpbm.T_Interaction)
        messages["mpbm.mpbmOutMsg.r_Slug_B"] = r_slug
        messages["mpbm.mpbmOutMsg.v_Slug_B"] = v_slug
        messages["mpbm.mpbmOutMsg.T_Interaction"] = torque_int

    return {"messages": messages}

def run_mc_generation(*, numRuns: int, ctrlDt: float, logDt: float, simDt: float, simTime: float, numThreads: int = 4, sloshingModel: str = "none", validation: bool = False):
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)
    global MC_CTRL_DT, MC_LOG_DT, MC_SIM_DT, MC_SIM_TIME, MC_SLOSHING_MODEL, MC_VALIDATION
    MC_CTRL_DT = ctrlDt
    MC_LOG_DT = logDt
    MC_SIM_DT = simDt
    MC_SIM_TIME = simTime
    MC_SLOSHING_MODEL = sloshingModel
    MC_VALIDATION = validation
    
    experimentBaseDir = os.path.join("data", "experiments", fileName)
    os.makedirs(experimentBaseDir, exist_ok=True)
    rawBaseDir = os.path.join("data", "raw")
    os.makedirs(rawBaseDir, exist_ok=True)
    
    datetimeStr = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    monteCarlo = Controller()
    monteCarlo.setSimulationFunction(createScenario)
    monteCarlo.setExecutionFunction(executeScenario)
    monteCarlo.setExecutionCount(numRuns)
    monteCarlo.setShouldDisperseSeeds(True)
    monteCarlo.setThreadCount(numThreads)
    monteCarlo.setVerbose(False)
    monteCarlo.setShowProgressBar(True)
    monteCarlo.setArchiveDir(os.path.join(experimentBaseDir, datetimeStr))
    
    retentionPolicy = RetentionPolicy()
    retentionPolicy.logRate = macros.sec2nano(logDt)
    retentionPolicy.addMessageLog("attError.attGuidOutMsg", ["sigma_BR", "omega_BR_B"])
    retentionPolicy.addMessageLog("rngControl.cmdTorqueOutMsg", ["torqueRequestBody"])
    if validation and sloshingModel == "mpbm":
        retentionPolicy.addMessageLog("mpbm.mpbmOutMsg", ["r_Slug_B", "v_Slug_B", "T_Interaction"])
    monteCarlo.addRetentionPolicy(retentionPolicy)
    
    print(f"Starting Monte Carlo simulation with {numRuns} runs...")
    failures = monteCarlo.executeSimulations()
    if failures:
        print(f"Failed runs reported by MC controller: {failures}")
        
    h5_filename = f"{sloshingModel}_{numRuns}_{simTime}_{logDt}.h5" if sloshingModel != "none" else f"attitude_{numRuns}_{simTime}_{logDt}.h5"
    h5_path = os.path.join(rawBaseDir, h5_filename)
    
    print(f"Saving data to {h5_path}...")
    with h5py.File(h5_path, 'w') as f:
        grp_ts = f.create_group("timeseries")
        for i in range(numRuns):
            data = monteCarlo.getRetainedData(i)
            retry_count = 0
            while check_for_nans(data) and retry_count < 10:
                print(f"WARNING: NaN detected in Run {i}. Retrying... (Attempt {retry_count + 1})")
                data = run_single_retry()
                retry_count += 1
            if check_for_nans(data):
                print(f"ERROR: Run {i} failed to produce clean data.")
            
            grp_seq = grp_ts.create_group(f"sequence_{i}")
            sigma_mrp = data["messages"]["attError.attGuidOutMsg.sigma_BR"][:, 1:]
            omega = data["messages"]["attError.attGuidOutMsg.omega_BR_B"][:, 1:]
            torque = data["messages"]["rngControl.cmdTorqueOutMsg.torqueRequestBody"][:, 1:]
            
            num_steps = sigma_mrp.shape[0]
            quaternion = np.zeros((num_steps, 4))
            for k in range(num_steps):
                quaternion[k] = rbk.MRP2EP(sigma_mrp[k])
            
            state = np.hstack((quaternion, omega))
            if torque.shape[0] < state.shape[0]:
                torque = np.vstack((torque, np.zeros((state.shape[0] - torque.shape[0], 3))))
            elif torque.shape[0] > state.shape[0]:
                torque = torque[:state.shape[0]]
            
            grp_seq.create_dataset("state", data=state, dtype=np.float32)
            grp_seq.create_dataset("control_torque", data=torque, dtype=np.float32)

            if validation and sloshingModel == "mpbm":
                r_slug = data["messages"]["mpbm.mpbmOutMsg.r_Slug_B"][:, 1:]
                v_slug = data["messages"]["mpbm.mpbmOutMsg.v_Slug_B"][:, 1:]
                torque_int = data["messages"]["mpbm.mpbmOutMsg.T_Interaction"][:, 1:]
                grp_seq.create_dataset("slug_position", data=r_slug, dtype=np.float32)
                grp_seq.create_dataset("slug_velocity", data=v_slug, dtype=np.float32)
                grp_seq.create_dataset("interaction_torque", data=torque_int, dtype=np.float32)
            
    print("Data generation complete.")
    if os.path.exists(monteCarlo.archiveDir):
        shutil.rmtree(monteCarlo.archiveDir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-runs', type=int, default=1000)
    parser.add_argument('--ctrl-dt', type=float, default=1.0)
    parser.add_argument('--log-dt', type=float, default=0.1)
    parser.add_argument('--sim-dt', type=float, default=0.01)
    parser.add_argument('--sim-time', type=float, default=1000.0)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--sloshing', type=str, default='none', choices=['none', 'spring', 'mpbm'])
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    
    print(f"Starting data generation with Runs:{args.num_runs}, CtrlDt:{args.ctrl_dt}s, Threads:{args.threads}")
    run_mc_generation(
        numRuns=args.num_runs, ctrlDt=args.ctrl_dt, logDt=args.log_dt, 
        simDt=args.sim_dt, simTime=args.sim_time, numThreads=args.threads,
        sloshingModel=args.sloshing, validation=args.validation
    )