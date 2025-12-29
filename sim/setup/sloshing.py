import numpy as np
from Basilisk.simulation import fuelTank
from Basilisk.simulation import linearSpringMassDamper

class SloshingModel:
    def setup(self, scObject, modelTag="FuelTank"):
        """
        Setup fuel tank and sloshing particles.
        """
        raise NotImplementedError

class NoSloshing(SloshingModel):
    def __init__(self, mass_init: float = 100.0, r_TB_B: list = [0.0, 0.0, 0.1]):
        self.mass_init = mass_init
        self.r_TB_B = r_TB_B

    def setup(self, scObject, modelTag="FuelTank"):
        tank = fuelTank.FuelTank()
        tankModel = fuelTank.FuelTankModelConstantVolume()
        tankModel.propMassInit = self.mass_init
        tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        tankModel.radiusTankInit = 0.5 

        tank.setTankModel(tankModel)
        tank.r_TB_B = [[self.r_TB_B[0]], [self.r_TB_B[1]], [self.r_TB_B[2]]]
        tank.nameOfMassState = "fuelTankMass"
        tank.updateOnly = True
        
        scObject.addStateEffector(tank)
        return tank

class SpringMassSloshing(SloshingModel):
    def __init__(self, 
                 mass_init: float, 
                 particles: list, 
                 r_TB_B: list = [0.0, 0.0, 0.1]):
        self.mass_init = mass_init
        self.particles_config = particles
        self.r_TB_B = r_TB_B

    def setup(self, scObject, modelTag="FuelTank"):
        tank = fuelTank.FuelTank()
        tankModel = fuelTank.FuelTankModelConstantVolume()
        tankModel.propMassInit = self.mass_init
        tankModel.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        tankModel.radiusTankInit = 0.5 

        tank.setTankModel(tankModel)
        tank.r_TB_B = [[self.r_TB_B[0]], [self.r_TB_B[1]], [self.r_TB_B[2]]]
        tank.nameOfMassState = "fuelTankMass"
        tank.updateOnly = True

        for p_cfg in self.particles_config:
            particle = linearSpringMassDamper.LinearSpringMassDamper()
            particle.k = p_cfg['k']
            particle.c = p_cfg['c']
            particle.massInit = p_cfg['mass']
            particle.rhoInit = p_cfg.get('rho', 0.0)
            particle.rhoDotInit = p_cfg.get('rho_dot', 0.0)
            
            pos = p_cfg['pos']
            direction = p_cfg['dir']
            particle.r_PB_B = [[pos[0]], [pos[1]], [pos[2]]]
            particle.pHat_B = [[direction[0]], [direction[1]], [direction[2]]]

            tank.pushFuelSloshParticle(particle)
            scObject.addStateEffector(particle)

        scObject.addStateEffector(tank)
        return tank

class MPBMSloshing(SloshingModel):
    def __init__(self, mass_init: float = 100.0):
        self.mass_init = mass_init

    def setup(self, scObject, modelTag="FuelTank"):
        # Placeholder for MPBM implementation
        # Assuming similar structure but different particle physics or effector
        tank = fuelTank.FuelTank()
        tankModel = fuelTank.FuelTankModelConstantVolume()
        tankModel.propMassInit = self.mass_init
        tank.setTankModel(tankModel)
        scObject.addStateEffector(tank)
        return tank

