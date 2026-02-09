class CargoPickupEnv:
    """MVP environment stub for cargo pickup."""

    def reset(self):
        return {}

    def step(self, action):
        return {}, 0.0, False, {}
