import logging
import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sim.setup.bsk_sim import BskSim

log = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="evaluate", version_base=None)
def run(cfg: DictConfig):
    log.info("\n" + "="*80)
    log.info("SIMULATION CONFIGURATION")
    log.info("="*80)
    log.info(OmegaConf.to_yaml(cfg))
    log.info("="*80 + "\n")

    log.info("Initializing BskSim...")
    sim = BskSim(cfg.sim, cfg.control)
    
    log.info("Initializing Simulation Processes...")
    sim.init_simulation()

    log.info(f"Executing Simulation for {cfg.sim.sim_time} seconds...")
    sim.run()

    log.info("Simulation Finished.")
    
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except Exception as e:
        log.error(f"Failed to get Hydra output directory: {e}")
        sys.exit(1)

    # Print Controller Profiling Stats
    if hasattr(sim.fsw.controller, "profiler"):
        # Use ModelTag if available, otherwise default to "Controller"
        controller_name = getattr(sim.fsw.controller, "ModelTag", "Controller")
        sim.fsw.controller.profiler.print_stats(name=controller_name)

    if cfg.sim.show_plots:
        log.info("Generating plots...")
        sim.plot_results(save_dir=output_dir)
        
    log.info("Saving data...")
    sim.save_data(save_dir=output_dir)

if __name__ == "__main__":
    run()
