from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from xmimic.controllers import ActionSemantics, PDController, PDGains


@dataclass
class IsaacEnvConfig:
    asset_root: str
    humanoid_urdf: str
    object_urdf: str
    num_envs: int = 1
    dt: float = 1.0 / 60.0
    substeps: int = 2
    use_gpu: bool = True
    action_mode: str = "target"
    action_scale: float = 1.0
    pd_kp: float = 100.0
    pd_kd: float = 10.0
    torque_limit: float = 150.0
    control_rate_hz: float = 100.0
    pd_rate_hz: float = 1000.0


class CargoPickupIsaacEnv:
    def __init__(self, cfg: IsaacEnvConfig):
        try:
            from isaacgym import gymapi
        except Exception as exc:
            raise ImportError("Isaac Gym is required for CargoPickupIsaacEnv") from exc

        self.cfg = cfg
        self.gym = gymapi.acquire_gym()
        self.gymapi = gymapi

        sim_params = gymapi.SimParams()
        sim_params.dt = cfg.dt
        sim_params.substeps = cfg.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = cfg.use_gpu

        compute_id = 0
        graphics_id = 0
        self.sim = self.gym.create_sim(compute_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False

        self.humanoid_asset = self.gym.load_asset(self.sim, cfg.asset_root, cfg.humanoid_urdf, asset_options)
        self.object_asset = self.gym.load_asset(self.sim, cfg.asset_root, cfg.object_urdf, asset_options)

        dof_count = self.gym.get_asset_dof_count(self.humanoid_asset)
        gains = PDGains(
            kp=np.full((dof_count,), cfg.pd_kp, dtype=np.float32),
            kd=np.full((dof_count,), cfg.pd_kd, dtype=np.float32),
            torque_limit=np.full((dof_count,), cfg.torque_limit, dtype=np.float32),
        )
        semantics = ActionSemantics(
            mode=cfg.action_mode,
            action_scale=cfg.action_scale,
            control_rate_hz=cfg.control_rate_hz,
            pd_rate_hz=cfg.pd_rate_hz,
        )
        self.pd = PDController(gains, semantics)
        self.dof_count = dof_count

        self.envs = []
        self.actors = []
        spacing = 2.0
        for idx in range(cfg.num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-spacing, 0.0, 0.0), gymapi.Vec3(spacing, spacing, spacing), cfg.num_envs)
            humanoid_handle = self.gym.create_actor(env, self.humanoid_asset, gymapi.Transform(), f"humanoid_{idx}", idx, 1)
            object_handle = self.gym.create_actor(env, self.object_asset, gymapi.Transform(), f"object_{idx}", idx, 2)
            self.envs.append(env)
            self.actors.append((humanoid_handle, object_handle))

    def reset(self) -> Dict[str, Any]:
        # Basic reset: zero all actor root states
        for env, (humanoid, obj) in zip(self.envs, self.actors):
            root_state = self.gym.get_actor_root_state(env, humanoid)
            root_state["pose"]["p"].x = 0.0
            root_state["pose"]["p"].y = 0.0
            root_state["pose"]["p"].z = 1.0
            self.gym.set_actor_root_state(env, humanoid, root_state)

            obj_state = self.gym.get_actor_root_state(env, obj)
            obj_state["pose"]["p"].x = 0.2
            obj_state["pose"]["p"].y = 0.0
            obj_state["pose"]["p"].z = 0.8
            self.gym.set_actor_root_state(env, obj, obj_state)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        return self._get_root_states()

    def step(self, actions) -> Dict[str, Any]:
        if actions is not None and self.dof_count > 0:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = np.repeat(actions[None, :], len(self.envs), axis=0)
            for env, (humanoid, _) , action in zip(self.envs, self.actors, actions):
                dof_state = self.gym.get_actor_dof_states(env, humanoid, self.gymapi.STATE_ALL)
                q = np.asarray(dof_state["pos"], dtype=np.float32)
                qd = np.asarray(dof_state["vel"], dtype=np.float32)
                torque = self.pd.compute_torque(q, qd, action)
                try:
                    self.gym.set_actor_dof_efforts(env, humanoid, torque)
                except Exception:
                    # Some Isaac builds only support position targets; ignore if efforts are unsupported.
                    pass
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        return self._get_root_states()

    def _get_root_states(self) -> Dict[str, Any]:
        states = []
        for env, (humanoid, obj) in zip(self.envs, self.actors):
            humanoid_state = self.gym.get_actor_root_state(env, humanoid)
            object_state = self.gym.get_actor_root_state(env, obj)
            states.append(
                {
                    "humanoid_pos": [
                        humanoid_state["pose"]["p"].x,
                        humanoid_state["pose"]["p"].y,
                        humanoid_state["pose"]["p"].z,
                    ],
                    "object_pos": [
                        object_state["pose"]["p"].x,
                        object_state["pose"]["p"].y,
                        object_state["pose"]["p"].z,
                    ],
                }
            )
        return {"root_states": states}
