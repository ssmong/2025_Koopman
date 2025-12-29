import hydra
from omegaconf import DictConfig

from Basilisk.simulation import spacecraft, gravityEffector
from Basilisk.utilities import orbitalMotion, simIncludeGravBody, rbk

def setup_sc_model(sc_cfg: DictConfig):
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = sc_cfg.tag

    scObject.hub.mHub = sc_cfg.mass
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(sc_cfg.inertia)

    ep_BN = sc_cfg.init_attitude.ep_BN
    sigma_BN = rbk.EP2MRP(ep_BN)
    scObject.hub.sigma_BNInit = sigma_BN
    scObject.hub.omega_BN_BInit = sc_cfg.init_attitude.omega_BN_B

    return scObject

def setup_env(env_cfg: DictConfig, oe: orbitalMotion.ClassicElements):
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravBodies = gravFactory.createBodies('earth', 'sun')

    earth = gravBodies['earth']
    earth.useSphericalHarmonicsGravityModel(
        bskPath + '/supportData/LocalGravData/GGM03S.txt', 4)
    
    mu = earth.mu
    rN, vN = orbitalMotion.elem2rv(mu, oe)
    scObject.hub.r_CN_NInit = rN
    scObject.hub.v_CN_NInit = vN

    gravEffector = gravityEffector.GravityEffector()
    gravEffector.ModelTag = "Gravity"
    