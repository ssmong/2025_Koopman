import os
import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

from Basilisk.utilities import SimulationBaseClass, macros, unitTestSupport, orbitalMotion
from Basilisk.utilities import simIncludeGravBody, vizSupport
from Basilisk.simulation import spacecraft, extForceTorque, simpleNav
from Basilisk.fswAlgorithms import attTrackingError, inertial3D
from Basilisk.architecture import messaging

from sim.controller.koopman_mpc import BskKoopmanMPC

log = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="evaluate.yaml", version_base=None)
def run(cfg: DictConfig):
    log.info("\n" + "="*80)
    log.info("SIMULATION CONFIGURATION")
    log.info("="*80)
    log.info(OmegaConf.to_yaml(cfg))
    log.info("="*80 + "\n")

    # -------------------------------------------------------------------------
    # 1. Simulation Setup
    # -------------------------------------------------------------------------
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    
    dynProcess = scSim.CreateNewProcess(simProcessName)
    simulationTimeStep = macros.sec2nano(cfg.simulation.step_size)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # -------------------------------------------------------------------------
    # 2. Plant (Spacecraft & Environment)
    # -------------------------------------------------------------------------
    
    # 2.1 Spacecraft
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "SMRL-Sat"
    
    # Mass Properties
    scObject.hub.mHub = cfg.simulation.spacecraft.mass
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(cfg.simulation.spacecraft.inertia)
    
    # Initial Orbit (Earth Centered)
    oe = orbitalMotion.ClassicElements()
    oe.a = cfg.simulation.spacecraft.orbit.a * 1000 # km -> m
    oe.e = cfg.simulation.spacecraft.orbit.e
    oe.i = cfg.simulation.spacecraft.orbit.i * macros.D2R
    oe.Omega = cfg.simulation.spacecraft.orbit.Omega * macros.D2R
    oe.omega = cfg.simulation.spacecraft.orbit.omega * macros.D2R
    oe.f = cfg.simulation.spacecraft.orbit.f * macros.D2R
    
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    
    # Initial Attitude
    scObject.hub.sigma_BNInit = cfg.simulation.spacecraft.initial_attitude.sigma_BN
    scObject.hub.omega_BN_BInit = cfg.simulation.spacecraft.initial_attitude.omega_BN_B

    scSim.AddModelToTask(simTaskName, scObject)

    # 2.2 Environment (Gravity)
    if cfg.simulation.environment.use_gravity:
        gravFactory = simIncludeGravBody.gravBodyFactory()
        earth = gravFactory.createEarth()
        earth.isCentralBody = True
        earth.useSphericalHarmonicsGravityModel(simIncludeGravBody.gravBodyFactory.path + '/GGM03S.txt', 4)
        gravFactory.addBodiesTo(scObject)
        scSim.AddModelToTask(simTaskName, gravFactory.spiceObject, 2)

    # 2.3 Navigation Sensor (SimpleNav)
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)
    
    # Noise Config
    sNavObject.PMatrix = np.zeros((18, 18)) # 공분산 행렬 (생략 가능하나 예시로 둠)
    
    # 2.4 Actuator Interface (External Torque)
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "ControlTorque"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject)

    # -------------------------------------------------------------------------
    # 3. FSW (Guidance & Control)
    # -------------------------------------------------------------------------

    # 3.1 Guidance: Inertial Pointing (Goal: 0,0,0)
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    inertial3DObj.sigma_R0N = [0., 0., 0.]
    scSim.AddModelToTask(simTaskName, inertial3DObj)

    # 3.2 Error Computation
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attError"
    scSim.AddModelToTask(simTaskName, attError)

    # 3.3 Controller: Koopman MPC
    #
    # Hydra Config에서 MPC 파라미터 로드
    mpc_params = OmegaConf.to_container(cfg.controller.mpc_params, resolve=True)
    
    # u_min/u_max가 스칼라로 들어올 경우 벡터로 변환 (KoopmanMPC 호환성)
    control_dim = 3
    if isinstance(mpc_params['u_min'], (int, float)):
        mpc_params['u_min'] = [mpc_params['u_min']] * control_dim
    if isinstance(mpc_params['u_max'], (int, float)):
        mpc_params['u_max'] = [mpc_params['u_max']] * control_dim

    print(f"[Sim] Instantiating BskKoopmanMPC with checkpoint: {cfg.controller.checkpoint_path}")
    
    # 체크포인트 경로 절대경로로 변환 (Hydra 실행 위치 문제 방지)
    checkpoint_path = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), cfg.controller.checkpoint_path))
    
    koopmanController = BskKoopmanMPC(
        checkpoint_dir=checkpoint_path,
        mpc_params=mpc_params,
        device=cfg.controller.device
    )
    koopmanController.ModelTag = "KoopmanMPC"
    scSim.AddModelToTask(simTaskName, koopmanController)

    # -------------------------------------------------------------------------
    # 4. Message Subscription
    # -------------------------------------------------------------------------
    
    # Sensor -> Error
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
    
    # Error -> Koopman Controller
    koopmanController.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    
    # Controller -> Actuator
    extFTObject.cmdTorqueInMsg.subscribeTo(koopmanController.cmdTorqueOutMsg)

    # -------------------------------------------------------------------------
    # 5. Logging & Vizard
    # -------------------------------------------------------------------------
    
    numDataPoints = int(cfg.simulation.duration / cfg.simulation.step_size)
    samplingTime = unitTestSupport.samplingTime(
        macros.sec2nano(cfg.simulation.duration), 
        simulationTimeStep, 
        numDataPoints
    )

    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    torqueLog = koopmanController.cmdTorqueOutMsg.recorder(samplingTime)
    
    scSim.AddModelToTask(simTaskName, attErrorLog)
    scSim.AddModelToTask(simTaskName, torqueLog)

    if cfg.simulation.use_vizard:
        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject)

    # -------------------------------------------------------------------------
    # 6. Execution
    # -------------------------------------------------------------------------
    
    print("[Sim] Initializing Basilisk Simulation...")
    scSim.InitializeSimulation()
    
    print(f"[Sim] Running for {cfg.simulation.duration} seconds...")
    scSim.ConfigureStopTime(macros.sec2nano(cfg.simulation.duration))
    scSim.ExecuteSimulation()
    print("[Sim] Finished.")

    # -------------------------------------------------------------------------
    # 7. Plotting
    # -------------------------------------------------------------------------
    
    if cfg.simulation.show_plots:
        timeAxis = attErrorLog.times() * macros.NANO2SEC
        sigma_BR = attErrorLog.sigma_BR
        omega_BR_B = attErrorLog.omega_BR_B
        cmdTorque = torqueLog.torqueRequestBody
        
        save_dir = os.getcwd() # Hydra가 생성한 output 폴더
        plot_results(timeAxis, sigma_BR, omega_BR_B, cmdTorque, save_dir)


def plot_results(time, sigma, omega, torque, save_dir):
    # Attitude Error
    plt.figure(1, figsize=(10, 6))
    for i in range(3):
        plt.plot(time, sigma[:, i], label=f'$\sigma_{i}$')
    plt.title('Attitude Error (MRP)')
    plt.ylabel('Sigma')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'attitude_error.png'))

    # Rate Error
    plt.figure(2, figsize=(10, 6))
    for i in range(3):
        plt.plot(time, omega[:, i], label=f'$\omega_{i}$')
    plt.title('Rate Tracking Error')
    plt.ylabel('Rate [rad/s]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'rate_error.png'))

    # Control Torque
    plt.figure(3, figsize=(10, 6))
    for i in range(3):
        plt.plot(time, torque[:, i], label=f'$u_{i}$')
    plt.title('Control Torque')
    plt.ylabel('Torque [Nm]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'control_torque.png'))
    
    print(f"[Sim] Plots saved to {save_dir}")
    # plt.show() # 서버 환경 등에서는 주석 처리

if __name__ == "__main__":
    run()