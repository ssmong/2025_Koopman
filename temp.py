import os

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from Basilisk import __path__
from Basilisk.architecture import messaging
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

from Basilisk.architecture.bskLogging import BasiliskError

from Basilisk.ExternalModules import randomTorque

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

def run(show_plots, liveStream, broadcastStream, 
        use2StarTracker, starTrackerFov, sunSensorFov, attitudeSetCase, 
        useNavError, useSloshing,
        controllerType, useRW,
        timeStep, simulationTime):    
    #
    #       실험 결과 저장 경로 설정
    #
    
    # data/experiments/{fileName}/{datetime}_{controllerType}/
    experimentBaseDir = os.path.join("data", "experiments")
    os.makedirs(experimentBaseDir, exist_ok=True)
    
    fileNameDir = os.path.join(experimentBaseDir, fileName)
    os.makedirs(fileNameDir, exist_ok=True)
    
    datetimeStr = datetime.now().strftime("%Y%m%d_%H%M%S")
    experimentDir = os.path.join(fileNameDir, f"{datetimeStr}_{controllerType}")
    os.makedirs(experimentDir, exist_ok=True)
    
    dataDir = os.path.join(experimentDir, "data")
    figureDir = os.path.join(experimentDir, "figure")
    os.makedirs(dataDir, exist_ok=True)
    os.makedirs(figureDir, exist_ok=True)

    #
    #
    #

    rngSeed = 42
    simTaskName = "simTask"
    simProcessName = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(False)

    #
    #       Simulation Plant 설정
    #

    dynProcess = scSim.CreateNewProcess(simProcessName)
    
    simulationTime = macros.sec2nano(simulationTime)
    simulationTimeStep = macros.sec2nano(timeStep)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))
    # 적분기는 필요 시 from Basilisk.simulation import svIntegrators 사용
    # 기본값은 RK4

    #
    #   1. Spacecraft 설정
    #
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "SMRL-Sat"
    # Spacecraft Parameters
    I = [101.67,  0.,      0., 
         0.,      135.42,  0.,
         0.,      0.,      153.75]
    scObject.hub.mHub = 500.0   # kg - spacecraft mass
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    # Spacecraft Initial COE
    oe = orbitalMotion.ClassicElements()
    oe.a = (6378.1366 + 402.72) * 1000      # meters
    oe.e = 0.00130547
    oe.i = 51.60 * macros.D2R
    oe.Omega = 198.38 * macros.D2R
    oe.omega = 39.26 * macros.D2R
    oe.f = 117.71 * macros.D2R
    # Spacecraft 위치, 속도는 mu 정의 후 추가
    #
    # Spacecraft Initial Attitude
    q_BN_start = [[0.648, 0.517, -0.414, 0.376]]    # [q0, q1, q2, q3], in basilisk style, q_BN = EP_BN
    q_BNInit = q_BN_start[attitudeSetCase]
    scObject.hub.sigma_BNInit = rbk.EP2MRP(q_BNInit)    # 기본은 MRP 사용
    scObject.hub.omega_BN_BInit = [[0.], [0.], [0.]]             # rad/s - omega_CN_B
    # Add spacecraft to simulation task
    scSim.AddModelToTask(simTaskName, scObject, 1)

    #
    #   2. 환경 설정
    #
    # 2.1. 중력장 설정
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies('earth', 'sun')  # 태양 추가 (SRP에 필요)
    earth = gravBodies['earth']
    earth.useSphericalHarmonicsGravityModel(
        bskPath + '/supportData/LocalGravData/GGM03S.txt', 4)  # J4까지 사용, 증가 가능
    mu = earth.mu
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN
    # Spacecraft가 중력장에 영향을 받도록 설정 (필수)
    gravFactory.addBodiesTo(scObject)
    #
    # 2.2. 태양 설정
    # SPICE 설정 (SRP를 위해 태양 위치 정보 필요 / SRP 사용 안 할 경우 Optional)
    timeInitString = '2025 DECEMBER 09 00:00:00.0'
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    spiceObject.zeroBase = 'Earth'  # 중력장 중심 지구로 설정
    scSim.AddModelToTask(simTaskName, gravFactory.spiceObject, 2)
    sunIdx = 1  # createBodies('earth', 'sun')에서 sun이 첫 번째, SRP에서 사용
    #
    # 2.3. 기체 모델 설정 (Exponential Atmosphere)
    # 필요 시 Tabular 혹은 MSIS로 변경
    atmo = exponentialAtmosphere.ExponentialAtmosphere()
    simSetPlanetEnvironment.exponentialAtmosphere(atmo, 'earth')

    #
    #   3. 외란 설정
    #
    # 3.1. 공기 저항
    dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
    # 3.1.1. 위성 본체 6면 패널
    hubDragCoeff = 2.2          # [-] (공기 저항 계수)
    hubSize = 1.0               # [m] (1m x 1m x 1m) 정육면체 위성 가정
    hubArea = hubSize ** 2.0    # [m^2] 위성 패널 면적
    hubOffSet = hubSize / 2.0   # [m] (패널 중심부터 위성 중심까지의 거리)
    hubNormals = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    for normal in hubNormals:
        location = normal * hubOffSet
        dragEffector.addFacet(hubArea, hubDragCoeff, normal, location)
    # 3.1.2. 위성 태양광 패널 (2m x 1m, y축 방향 2개 가정)
    panelArea, panelCoeff = 2.0, 2.2
    panelDist = hubOffSet + 1.0  # 본체 절반 + 패널 길이 절반
    # 패널 데이터 (+Y 날개와 -Y 날개의 앞면(+Z)/뒷면(-Z) 총 4개 면)
    panel_data = [
        (np.array([0, 0,  1]), np.array([0,  panelDist, 0])), # +Y 날개, +Z면
        (np.array([0, 0, -1]), np.array([0,  panelDist, 0])), # +Y 날개, -Z면
        (np.array([0, 0,  1]), np.array([0, -panelDist, 0])), # -Y 날개, +Z면
        (np.array([0, 0, -1]), np.array([0, -panelDist, 0]))  # -Y 날개, -Z면
    ]
    for normal, loc in panel_data:
        dragEffector.addFacet(panelArea, panelCoeff, normal, loc)
    # Spacecraft에 공기 저항 외란 추가
    scObject.addDynamicEffector(dragEffector)
    #
    # 3.2. 태양 Radiation Pressure (FacetSRP)
    srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
    srpEffector.ModelTag = "FacetSRP"
    # Facet 개수 설정 (본체 6면 + 패널 4면 = 10개)
    numFacets = 10
    numArticulatedFacets = 0  # 회전 없음
    srpEffector.setNumFacets(numFacets)
    srpEffector.setNumArticulatedFacets(numArticulatedFacets)
    # 태양 메시지 연결
    srpEffector.sunInMsg.subscribeTo(gravFactory.spiceObject.planetStateOutMsgs[sunIdx])
    # Facet 정보 정의
    # F frame: 각 facet에 고정된 좌표계 (Facet frame), +Y 축이 법선 방향
    # 회전하지 않는 facet이지만 모듈 API상 dcm_F0B 필요

    # 3.2.1. 위성 본체 6면
    hubDiffuseCoeff = 0.1
    hubSpecularCoeff = 0.9
    for normal in hubNormals:
        location = normal * hubOffSet
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])  # F frame +Y = 법선
        rotHat_F = np.array([0.0, 0.0, 0.0])  # 회전 없음
        srpEffector.addFacet(hubArea, dcm_F0B, nHat_F, rotHat_F, location, 
                            hubDiffuseCoeff, hubSpecularCoeff)
    # 3.2.2. 태양광 패널 4면
    panelDiffuseCoeff = 0.16
    panelSpecularCoeff = 0.16
    for normal, loc in panel_data:
        dcm_F0B = normalToDcmF0B(normal)
        nHat_F = np.array([0.0, 1.0, 0.0])
        rotHat_F = np.array([0.0, 0.0, 0.0])
        srpEffector.addFacet(panelArea, dcm_F0B, nHat_F, rotHat_F, loc,
                            panelDiffuseCoeff, panelSpecularCoeff)    
    # Spacecraft에 SRP 외란 추가
    scObject.addDynamicEffector(srpEffector)
    scSim.AddModelToTask(simTaskName, srpEffector)
    #
    # 3.3. 중력 gradient descent 토크 외란
    ggEffector = GravityGradientEffector.GravityGradientEffector()
    ggEffector.ModelTag = "GravityGradient"
    ggEffector.addPlanetName(earth.planetName)
    scObject.addDynamicEffector(ggEffector)
    scSim.AddModelToTask(simTaskName, ggEffector)
    #
    # 3.4. 슬로싱 외란 (Optional)
    # Fuel Tank 생성 및 설정
    tank = fuelTank.FuelTank()
    tankModel = fuelTank.FuelTankModelConstantVolume()
    tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
    tankModel.radiusTankInit = 0.5
    particles = []
    if useSloshing:
        directions = [[1,0,0], [0,1,0], [0,0,1]]
        positions = [[0.1,0,-0.1], [0,0,0.1], [-0.1,0,0.1]]
        # x,y,z축 별 슬로싱 파티클 생성
        for i, (direction, position) in enumerate(zip(directions, positions)):
            particle = linearSpringMassDamper.LinearSpringMassDamper()
            particle.k = 100.0
            particle.c = 5.0  # 댐핑
            particle.r_PB_B = [[position[0]], [position[1]], [position[2]]]
            particle.pHat_B = [[direction[0]], [direction[1]], [direction[2]]]
            particle.rhoInit = 0.05 if i == 0 else -0.025
            particle.rhoDotInit = 0.0
            particle.massInit = 10.0
            particles.append(particle)
        # 슬로싱 사용 시: particles (30kg) + propMassInit (70kg) = 총 100kg
        tankModel.propMassInit = 70.0       # 정지 질량 (추력기 사용 시 소모)
    else:
        # 슬로싱 고려 안 함: FuelTank만 사용 (particles 없이)
        # 총 연료 질량 = particles (30kg) + propMassInit (70kg) = 100kg
        tankModel.propMassInit = 100.0      # 정지 질량 (추력기 사용 시 소모)
    # TankModel 설정 및 Tank 속성 설정
    tank.setTankModel(tankModel)
    tank.r_TB_B = [[0], [0], [0.1]]
    tank.nameOfMassState = "fuelTankMass"
    tank.updateOnly = True
    # Particles를 Tank에 추가 및 Spacecraft에 추가
    for particle in particles:
        tank.pushFuelSloshParticle(particle)
        scObject.addStateEffector(particle)
    # Spacecraft에 Tank 추가
    scObject.addStateEffector(tank)
        

    #
    #   4. 자세 제어기 설정 (Ideal 가정)
    #
    # 필요 시 리액션휠이나 추력기로 변경
    if useRW:
        rwFactory = simIncludeRW.rwFactory()
        varRWModel = messaging.BalancedWheels
        RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100.  # RPM
                           , RWModel=varRWModel
                           )
        RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200.  # RPM
                           , RWModel=varRWModel
                           )
        RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300.  # RPM
                           , RWModel=varRWModel
                           )
        numRW = rwFactory.getNumOfDevices()
        rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwStateEffector.ModelTag = "RW_cluster"
        rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
        scSim.AddModelToTask(simTaskName, rwStateEffector, 2)
        # Torque -> RW Motor Torque 매핑 담당
        rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
        rwMotorTorqueObj.ModelTag = "RWMotorTorque"
        scSim.AddModelToTask(simTaskName, rwMotorTorqueObj)
        # RW가 body 3축 모두 제어하도록 설정
        controlAxes_B = [
            1, 0, 0, 0, 1, 0, 0, 0, 1
        ]
        rwMotorTorqueObj.controlAxes_B = controlAxes_B
    else:
        ctrlFTObject = extForceTorque.ExtForceTorque()
        ctrlFTObject.ModelTag = "controlForceTorque"
        scObject.addDynamicEffector(ctrlFTObject)
        scSim.AddModelToTask(simTaskName, ctrlFTObject)

    #
    #   5. 네비게이션 설정
    #
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)
    if useNavError:
        sigmaErrorSigma = 0.05 * macros.D2R
        omegaErrorSigma = 0.005 * macros.D2R
        # 노이즈 공분산 행렬의 제곱근
        # [0-2]   : 위치 오차 (r_BN_N)
        # [3-5]   : 속도 오차 (v_BN_N)
        # [6-8]   : 자세 오차 (sigma_BN) ← 자세 제어에 중요!
        # [9-11]  : 각속도 오차 (omega_BN_B)
        # [12-14] : 태양 방향 오차 (vehSunPntBdy)
        # [15-17] : 누적 속도 변화 오차 (vehAccumDV)
        PMatrix = np.zeros((18, 18))
        errorBounds = np.zeros(18)

        idx_sigma = slice(6, 9)
        idx_omega = slice(9, 12)
        PMatrix[range(6, 9), range(6, 9)] = sigmaErrorSigma
        PMatrix[range(9, 12), range(9, 12)] = omegaErrorSigma
        errorBounds[idx_sigma] = 3 * sigmaErrorSigma
        errorBounds[idx_omega] = 3 * omegaErrorSigma

        sNavObject.PMatrix = PMatrix
        sNavObject.walkBounds = errorBounds
        sNavObject.RNGSeed = rngSeed

    # 
    #       FSW 알고리즘 설정
    #

    # 목표 자세 설정
    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DObj)
    inertial3DObj.sigma_R0N = [0., 0., 0.] 

    # 자세 오차 계산 모듈
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)
    
    # 랜덤 토크 생성 모듈 (Temporary)
    if controllerType == "random":
        torqueControl = randomTorque.RandomTorque()
        torqueControl.ModelTag = "randomTorque"
        scSim.AddModelToTask(simTaskName, torqueControl)
        torqueControl.setTorqueMagnitude(1)    # [Nm]
        torqueControl.setSeed(rngSeed)
    else:
        raise BasiliskError(f"Invalid controller type: {controllerType}")

    # 
    #       Data Logging & Message 설정
    #
    numDataPoints = 101
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
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

    vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=I)
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


    #
    #       Vizard Visualization 설정
    #
    useVizard = (liveStream or broadcastStream) and vizSupport.vizFound
    if useVizard:
        clockSync = simSynch.ClockSynch()
        clockSync.accelFactor = 5.0
        scSim.AddModelToTask(simTaskName, clockSync)
        viz = vizSupport.enableUnityVisualization(
            scSim, simTaskName, scObject,
            rwEffectorList=(rwStateEffector if useRW else None),
            liveStream=liveStream,
            broadcastStream=broadcastStream,
            )
        viz.settings.showRWLabels = 0  
        viz.settings.viewRWHUD = 1
        # Keep In/Out Cone
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
        
        
    # 
    #       Simulation 실행
    #
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
        
    # 
    #       결과 Plot
    #
    # retrieve the logged data
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
    dataTorque = torqueLog.torqueRequestBody
    timeAxis = attErrorLog.times() * macros.NANO2SEC 
    # Plot 1: Attitude Error
    plot_attitude_error(timeAxis, dataSigmaBR, position=(100, 100))
    pltName = "attitude_error.png"
    plt.savefig(os.path.join(figureDir, pltName), dpi=150, bbox_inches='tight')
    
    # Plot 2: Rate Error
    plot_rate_error(timeAxis, dataOmegaBR, position=(100, 800))
    pltName = "rate_error.png"
    plt.savefig(os.path.join(figureDir, pltName), dpi=150, bbox_inches='tight')
    
    # Plot 3: Control Torque
    plot_torque(timeAxis, dataTorque, position=(1100, 100))
    pltName = "torque.png"
    plt.savefig(os.path.join(figureDir, pltName), dpi=150, bbox_inches='tight')
    
    if useRW:
        dataRWMotorTorque = rwMotorTorqueLog.motorTorque
        dataRWSpeed = rwStateLog.wheelSpeeds
        
        # Plot 4: RW Motor Torque
        plot_rw_motor_torque(timeAxis, dataRWMotorTorque, position=(100, 500))
        pltName = "rw_motor_torque.png"
        plt.savefig(os.path.join(figureDir, pltName), dpi=150, bbox_inches='tight')
        
        # Plot 5: RW Speed
        plot_rw_speeds(timeAxis, dataRWSpeed, position=(1100, 500))
        pltName = "rw_speed.png"
        plt.savefig(os.path.join(figureDir, pltName), dpi=150, bbox_inches='tight')

    # 모든 그래프를 한 번에 표시
    if show_plots:
        plt.show()
    
    # 모든 창 닫기
    plt.close('all')
    return

def normalToDcmF0B(nHat_B):
        """B frame 법선 벡터로부터 dcm_F0B 계산 (F frame +Y = nHat_B)"""
        nHat_B = np.array(nHat_B) / np.linalg.norm(nHat_B)
        if np.allclose(nHat_B, [0, 1, 0]):
            return np.eye(3)
        elif np.allclose(nHat_B, [0, -1, 0]):
            return rbk.PRV2C(np.pi * np.array([1.0, 0.0, 0.0]))
        else:
            # Gram-Schmidt로 직교 기저 구성
            y_axis = nHat_B
            temp = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
            x_axis = temp - np.dot(temp, y_axis) * y_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis) / np.linalg.norm(np.cross(x_axis, y_axis))
            return np.column_stack([x_axis, y_axis, z_axis]).T

# Plotting functions
def plot_attitude_error(timeData, dataSigmaBR, position=(100, 100)):
    """Plot the attitude errors."""
    fig = plt.figure(1, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    for idx in range(3):
        plt.plot(timeData, dataSigmaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')
    plt.grid(True)
    plt.title('Attitude Error')

def plot_rate_error(timeData, dataOmegaBR, position=(100, 600)):
    """Plot the body angular velocity rate tracking errors."""
    fig = plt.figure(2, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    for idx in range(3):
        plt.plot(timeData, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Rate Tracking Error (rad/s)')
    plt.grid(True)
    plt.title('Rate Tracking Error')

def plot_torque(timeData, dataTorque, position=(900, 100)):
    """Plot the control torque."""
    fig = plt.figure(3, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass
    
    for idx in range(3):
        plt.plot(timeData, dataTorque[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\tau_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Torque (Nm)')
    plt.grid(True)
    plt.title('Control Torque')

def plot_rw_motor_torque(timeData, dataUsReq, position=(100, 500)):
    """Plot the RW motor torques."""
    fig = plt.figure(4, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass

    for idx in range(3):
        plt.plot(timeData, dataUsReq[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='$u_{s,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('RW Motor Torque [Nm]')
    plt.grid(True)
    plt.title('RW Motor Torque')

def plot_rw_speeds(timeData, dataOmegaRW, position=(1100, 500)):
    """Plot the RW speeds."""
    fig = plt.figure(5, figsize=(10, 6))
    try:
        mngr = plt.get_current_fig_manager()
        if hasattr(mngr, 'window'):
            mngr.window.wm_geometry(f"+{position[0]}+{position[1]}")
    except:
        pass

    for idx in range(3):
        plt.plot(timeData, dataOmegaRW[:, idx] * macros.RPM,
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\Omega_{' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('RW Speed [RPM]')
    plt.grid(True)
    plt.title('RW Speeds')

if __name__ == "__main__":
    run(
        True,                       # show_plots
        True,                       # liveStream
        False,                      # broadcastStream
        False,                      # use2StarTracker
        starTrackerFov=20,          # starTrackerFov
        sunSensorFov=50,            # sunSensorFov
        attitudeSetCase=0,          # attitudeSetCase
        useNavError=False,          # useNavError
        useSloshing=False,          # useSloshing
        controllerType="random",    # controllerType
        useRW=True,                 # useRW
        timeStep=1,              # timeStep
        simulationTime=100.0,       # simulationTime
    )