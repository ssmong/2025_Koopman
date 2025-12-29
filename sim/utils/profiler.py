import numpy as np

class ControllerProfiler:
    def __init__(self, skip_first=True):
        self.times = []
        self.solver_times = []
        self.skip_first = skip_first
        self.first_call = True

    def update(self, wall_time, solver_time=None):
        if self.skip_first and self.first_call:
            self.first_call = False
            return
        self.times.append(wall_time)
        if solver_time is not None:
            self.solver_times.append(solver_time)

    def get_stats(self):
        if not self.times:
            return None
            
        stats = {
            "wall_mean": np.mean(self.times),
            "wall_max": np.max(self.times),
            "wall_min": np.min(self.times),
            "count": len(self.times)
        }
        
        if self.solver_times:
            stats.update({
                "solver_mean": np.mean(self.solver_times),
                "solver_max": np.max(self.solver_times),
            })
            
        return stats
    
    def print_stats(self, name="Controller"):
        stats = self.get_stats()
        if not stats:
            print(f"[{name}] No profiling data collected.")
            return

        print(f"\n" + "="*40)
        print(f"[{name} Profiling Results]")
        print(f"  Count     : {stats['count']}")
        print(f"  Wall Mean : {stats['wall_mean']*1000:.4f} ms")
        print(f"  Wall WCET : {stats['wall_max']*1000:.4f} ms")
        
        if "solver_mean" in stats:
            print(f"  Solver Mean: {stats['solver_mean']*1000:.4f} ms")
            print(f"  Solver Max : {stats['solver_max']*1000:.4f} ms")
        print("="*40 + "\n")

