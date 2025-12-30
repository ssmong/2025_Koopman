import numpy as np
from Basilisk.simulation import (
    exponentialAtmosphere,
    facetDragDynamicEffector,
    facetSRPDynamicEffector,
)
from Basilisk.utilities import (
    simIncludeGravBody,
)
import Basilisk

class Environment:
    def __init__(self, bsk_sim):
        self.bsk_sim = bsk_sim
        self.env_cfg = bsk_sim.sim_cfg.env
        self.geom_cfg = bsk_sim.sim_cfg.sc.geometry
        
        self._setup_gravity()
        if self.env_cfg.enable_drag:
            self._setup_atmosphere()
            self._setup_drag()
        if self.env_cfg.enable_srp:
            self._setup_srp()

    def _setup_gravity(self):
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        gravBodies = self.gravFactory.createBodies('earth', 'sun')
        self.sunIdx = 1

        earth = gravBodies['earth']
        
        bskPath = Basilisk.__path__[0]
        earth.useSphericalHarmonicsGravityModel(
            bskPath + '/supportData/LocalGravData/GGM03S.txt', 4)

        self.mu = earth.mu
        
        self.gravFactory.addBodiesTo(self.bsk_sim.scObject)
        
        timeInitString = '2025 DECEMBER 09 00:00:00.0'
        self.spiceObject = self.gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
        self.spiceObject.zeroBase = 'Earth'
        
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, self.spiceObject, 2)
        
    def _setup_atmosphere(self):
        self.atmo = exponentialAtmosphere.ExponentialAtmosphere()
        self.atmo.ModelTag = "ExpAtmo"
        
        from Basilisk.utilities import simSetPlanetEnvironment
        simSetPlanetEnvironment.exponentialAtmosphere(self.atmo, 'earth')
        
        self.atmo.addSpacecraftToModel(self.bsk_sim.scObject.scStateOutMsg)
        
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, self.atmo, 2)

    def _get_geometry_data(self):
        hub_cfg = self.geom_cfg.hub
        panel_cfg = self.geom_cfg.panel
        
        hubSize = hub_cfg.size
        hubOffSet = hubSize / 2.0
        hubNormals = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        
        panelOffset = panel_cfg.offset
        panel_data = [
            (np.array([0, 0,  1]), np.array([0,  panelOffset, 0])),
            (np.array([0, 0, -1]), np.array([0,  panelOffset, 0])),
            (np.array([0, 0,  1]), np.array([0, -panelOffset, 0])),
            (np.array([0, 0, -1]), np.array([0, -panelOffset, 0]))
        ]
        
        return hub_cfg, hubNormals, hubOffSet, panel_cfg, panel_data

    def _setup_drag(self):
        dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        dragEffector.ModelTag = "FacetDrag"
        
        dragEffector.atmoDensInMsg.subscribeTo(self.atmo.envOutMsgs[0])

        hub_cfg, hubNormals, hubOffSet, panel_cfg, panel_data = self._get_geometry_data()
        
        for normal in hubNormals:
            location = normal * hubOffSet
            dragEffector.addFacet(hub_cfg.area, hub_cfg.coeff, normal, location)

        for normal, loc in panel_data:
            dragEffector.addFacet(panel_cfg.area, panel_cfg.coeff, normal, loc)

        self.bsk_sim.scObject.addDynamicEffector(dragEffector)
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, dragEffector, 2)

    def _setup_srp(self):
        srpEffector = facetSRPDynamicEffector.FacetSRPDynamicEffector()
        srpEffector.ModelTag = "FacetSRP"
        
        srpEffector.sunInMsg.subscribeTo(self.spiceObject.planetStateOutMsgs[self.sunIdx])
        
        hub_cfg, hubNormals, hubOffSet, panel_cfg, panel_data = self._get_geometry_data()
        
        for normal in hubNormals:
            location = normal * hubOffSet
            dcm_F0B = self.normalToDcmF0B(normal)
            nHat_F = np.array([0.0, 1.0, 0.0])
            rotHat_F = np.array([0.0, 0.0, 0.0])
            srpEffector.addFacet(hub_cfg.area, dcm_F0B, nHat_F, rotHat_F, location, 
                                hub_cfg.diff, hub_cfg.spec)
                                
        for normal, loc in panel_data:
            dcm_F0B = self.normalToDcmF0B(normal)
            nHat_F = np.array([0.0, 1.0, 0.0])
            rotHat_F = np.array([0.0, 0.0, 0.0])
            srpEffector.addFacet(panel_cfg.area, dcm_F0B, nHat_F, rotHat_F, loc,
                                panel_cfg.diff, panel_cfg.spec)
                                
        self.bsk_sim.scObject.addDynamicEffector(srpEffector)
        self.bsk_sim.scSim.AddModelToTask(self.bsk_sim.simTaskName, srpEffector, 2)

    @staticmethod
    def normalToDcmF0B(nHat_B):
        from Basilisk.utilities import RigidBodyKinematics as rbk
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
