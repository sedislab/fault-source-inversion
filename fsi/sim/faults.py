"""The fault catalogue: the space of faults we inject, grounded in the robot-FDI literature and mapped to the
Isaac Lab mechanism that realizes each one. This module is pure Python (no simulator) so the fault space, its
node-type applicability, and the sampling are testable on their own; sim/inject.py turns a FaultSpec into an
actual intervention on an Isaac Lab environment."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from ..core.graph import StructuralGraph, NodeType
from ..core.types import Onset

class FaultKind(str, Enum):
    TORQUE_LOSS = "torque_loss"
    LOCKED_JOINT = "locked_joint"
    FREE_SWING = "free_swing"
    GAIN_DEGRADE = "gain_degrade"
    FRICTION = "friction"
    BACKLASH = "backlash"
    DELAY = "delay"
    TORQUE_SATURATION = "torque_saturation"
    ROTOR_THRUST_LOSS = "rotor_thrust_loss"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_NOISE = "sensor_noise"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_FREEZE = "sensor_freeze"
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_SCALE = "sensor_scale"
    SENSOR_SPOOF = "sensor_spoof"
    MASS_CHANGE = "mass_change"
    COM_SHIFT = "com_shift"
    DAMPING = "damping"
    LIMB_LOSS = "limb_loss"
    EXTERNAL_WRENCH = "external_wrench"
    PERCEPTION_OOD = "perception_ood"
    DETECTION_MISS = "detection_miss"
    PREDICTION_ERROR = "prediction_error"
    PLANNER_FAILURE = "planner_failure"

@dataclass(frozen=True)
class FaultEntry:
    """One catalogue entry: which node types it applies to, its severity range, the Isaac Lab mechanism, and
    whether that mechanism is built in or needs our custom code."""
    kind: FaultKind
    node_types: tuple
    severity: tuple
    mechanism: str
    builtin: bool
    note: str = ""

_E = FaultEntry
_A, _S, _L, _B, _SW = NodeType.ACTUATOR, NodeType.SENSOR, NodeType.LINK, NodeType.BODY, NodeType.SOFTWARE

CATALOGUE_ENTRIES = [
    _E(FaultKind.TORQUE_LOSS, (_A,), (0.1, 0.9), "scale effort_limit / saturation_effort", False, "explicit actuator or per-step effort clamp"),
    _E(FaultKind.LOCKED_JOINT, (_A,), (1.0, 1.0), "stiff drive holding q_frozen", False, "no lock API; custom hold event"),
    _E(FaultKind.FREE_SWING, (_A,), (1.0, 1.0), "effort_limit ~ 0", True, "limp, not locked"),
    _E(FaultKind.GAIN_DEGRADE, (_A,), (0.2, 0.8), "randomize_actuator_gains", True, "explicit-actuator caveat"),
    _E(FaultKind.FRICTION, (_A, _L), (1.5, 6.0), "randomize_joint_parameters friction/armature", True, ""),
    _E(FaultKind.BACKLASH, (_A,), (0.02, 0.15), "custom deadband on joint target", False, ""),
    _E(FaultKind.DELAY, (_A,), (2, 12), "DelayedPDActuatorCfg min/max_delay", True, "steps of delay"),
    _E(FaultKind.TORQUE_SATURATION, (_A,), (0.3, 0.7), "lower saturation_effort", True, ""),
    _E(FaultKind.ROTOR_THRUST_LOSS, (_A,), (0.2, 1.0), "zero/scale a column of the rotor mixer", False, "custom 4-rotor mixer"),
    _E(FaultKind.SENSOR_BIAS, (_S,), (0.5, 3.0), "NoiseModelWithAdditiveBiasCfg", True, "per-episode bias"),
    _E(FaultKind.SENSOR_NOISE, (_S,), (2.0, 8.0), "GaussianNoiseCfg", True, "obs-layer noise"),
    _E(FaultKind.SENSOR_DRIFT, (_S,), (0.5, 3.0), "custom accumulating-ramp NoiseModel", False, ""),
    _E(FaultKind.SENSOR_FREEZE, (_S,), (1.0, 1.0), "custom latch-previous NoiseModel", False, ""),
    _E(FaultKind.SENSOR_DROPOUT, (_S,), (0.1, 0.6), "custom Bernoulli-mask NoiseModel", False, "dropout prob"),
    _E(FaultKind.SENSOR_SCALE, (_S,), (0.5, 2.0), "custom scale on obs term", False, ""),
    _E(FaultKind.SENSOR_SPOOF, (_S,), (1.0, 5.0), "custom offset injection (nav)", False, "multipath/spoofing"),
    _E(FaultKind.MASS_CHANGE, (_L, _B), (0.5, 2.0), "randomize_rigid_body_mass", True, ""),
    _E(FaultKind.COM_SHIFT, (_L, _B), (0.02, 0.12), "randomize_rigid_body_com", True, "meters"),
    _E(FaultKind.DAMPING, (_L, _A), (0.3, 3.0), "randomize joint damping", True, ""),
    _E(FaultKind.LIMB_LOSS, (_L,), (1.0, 1.0), "mass~0 + effort~0 on limb joints", False, "no edge deletion"),
    _E(FaultKind.EXTERNAL_WRENCH, (_L, _B), (1.0, 8.0), "apply_external_force_torque", True, "strongest fit"),
    _E(FaultKind.PERCEPTION_OOD, (_SW,), (1.0, 5.0), "OOD input -> nonconformity residual", False, "vehicle"),
    _E(FaultKind.DETECTION_MISS, (_SW,), (0.2, 0.8), "drop/perturb detections", False, "vehicle"),
    _E(FaultKind.PREDICTION_ERROR, (_SW,), (1.0, 5.0), "perturb prediction module output", False, "vehicle"),
    _E(FaultKind.PLANNER_FAILURE, (_SW,), (1.0, 5.0), "corrupt planned trajectory", False, "vehicle"),
]

@dataclass
class FaultSpec:
    """A concrete fault to inject: what kind, at which node, how severe, and its temporal profile."""
    kind: FaultKind
    node: int
    node_name: str
    severity: float
    onset: Onset = Onset.ABRUPT
    onset_frac: float = 0.4
    duration_frac: float = 1.0
    meta: dict = field(default_factory=dict)

class FaultCatalogue:
    def __init__(self, entries=None):
        self.entries = {e.kind: e for e in (entries or CATALOGUE_ENTRIES)}

    def applicable(self, graph: StructuralGraph, node: int) -> list[FaultKind]:
        t = graph.node(node).type
        return [k for k, e in self.entries.items() if t in e.node_types]

    def valid_targets(self, graph: StructuralGraph) -> list[tuple[int, FaultKind]]:
        return [(v, k) for v in graph.fail_nodes for k in self.applicable(graph, v)]

    def sample(self, graph: StructuralGraph, rng, node: int | None = None, kind: FaultKind | None = None) -> FaultSpec:
        if node is None:
            node = int(rng.choice(graph.fail_nodes))
        opts = self.applicable(graph, node)
        if not opts:
            raise ValueError(f"node {node} ({graph.node(node).type}) has no applicable faults")
        kind = kind or opts[int(rng.integers(len(opts)))]
        lo, hi = self.entries[kind].severity
        onset = Onset.ABRUPT if rng.random() < 0.6 else Onset.INCIPIENT
        return FaultSpec(kind, node, graph.node(node).name, float(rng.uniform(lo, hi)), onset,
                         float(rng.uniform(0.25, 0.55)))

    def sample_k(self, graph: StructuralGraph, rng, k: int) -> list[FaultSpec]:
        nodes = list(rng.choice(graph.fail_nodes, size=min(k, len(graph.fail_nodes)), replace=False))
        return [self.sample(graph, rng, node=int(v)) for v in nodes]

CATALOGUE = FaultCatalogue()
