from Basilisk.ExternalModules import randomTorque

class RandomController(randomTorque.RandomTorque):
    def __init__(self, torque_magnitude: float = 1.0, seed: int = 42):
        super().__init__()
        self.ModelTag = "randomTorque"
        self.setTorqueMagnitude(torque_magnitude)
        self.setSeed(seed)
