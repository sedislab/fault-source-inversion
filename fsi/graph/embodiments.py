"""The four structural graphs, matching the paper's figures: a serial manipulator, a quadruped, a multirotor,
and a heterogeneous autonomous vehicle that fuses sensors, software modules, and actuators in one graph. Node
ids are assigned in build order; these graphs are the fault->component map the data pipeline and models bind to."""
from __future__ import annotations
from ..core.graph import StructuralGraph, Node, Edge, NodeType, EdgeKind

class _Builder:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.edges = []
        self.idx = {}
    def add(self, name, type, **kw):
        i = len(self.nodes)
        self.nodes.append(Node(i, name, type, **kw))
        self.idx[name] = i
        return i
    def link(self, a, b, kind=EdgeKind.PHYSICAL, directed=False):
        self.edges.append(Edge(self.idx[a], self.idx[b], kind, directed))
    def build(self):
        return StructuralGraph(self.nodes, self.edges, self.name)

def franka_graph() -> StructuralGraph:
    """Franka Panda: 7 revolute actuators alternating with links, each joint a torque sensor, a wrist force sensor."""
    b = _Builder("franka")
    b.add("base", NodeType.LINK)
    prev = "base"
    for j in range(1, 8):
        b.add(f"J{j}", NodeType.ACTUATOR); b.link(prev, f"J{j}")
        b.add(f"tau{j}", NodeType.SENSOR); b.link(f"J{j}", f"tau{j}")
        if j < 7:
            b.add(f"L{j}", NodeType.LINK); b.link(f"J{j}", f"L{j}"); prev = f"L{j}"
        else:
            b.add("hand", NodeType.LINK); b.link("J7", "hand")
            b.add("wrist_ft", NodeType.SENSOR); b.link("hand", "wrist_ft")
    return b.build()

def anymal_graph() -> StructuralGraph:
    """ANYmal-style quadruped: base with IMU, four legs of hip-abduction/flexion/knee actuators ending in a foot."""
    b = _Builder("anymal")
    b.add("base", NodeType.BODY)
    b.add("imu", NodeType.SENSOR); b.link("base", "imu")
    for leg in ("LF", "RF", "LH", "RH"):
        for k, act in enumerate((f"{leg}_HAA", f"{leg}_HFE", f"{leg}_KFE")):
            b.add(act, NodeType.ACTUATOR)
            b.link("base" if k == 0 else prev, act)
            prev = act
        b.add(f"{leg}_foot", NodeType.SENSOR); b.link(prev, f"{leg}_foot")
    return b.build()

def crazyflie_graph() -> StructuralGraph:
    """Crazyflie multirotor: body with IMU and a height sensor, four rotor actuators coupled by control allocation."""
    b = _Builder("crazyflie")
    b.add("body", NodeType.BODY)
    b.add("imu", NodeType.SENSOR); b.link("body", "imu")
    b.add("baro", NodeType.SENSOR); b.link("body", "baro")
    for r in range(1, 5):
        b.add(f"M{r}", NodeType.ACTUATOR); b.link("body", f"M{r}", EdgeKind.ALLOCATION)
    return b.build()

def vehicle_graph() -> StructuralGraph:
    """Autonomous vehicle (CARLA): three strata in one graph. Sensors feed software modules; software drives
    physical actuators; actuators move the vehicle body, which the sensors observe (closing the loop)."""
    b = _Builder("vehicle")
    for s in ("cam", "lidar", "radar", "gnss", "imu"):
        b.add(s, NodeType.SENSOR)
    for m in ("perception", "fusion", "prediction", "planning", "control"):
        b.add(m, NodeType.SOFTWARE)
    for a in ("steer", "throttle", "brake"):
        b.add(a, NodeType.ACTUATOR)
    b.add("vehicle_body", NodeType.BODY)
    df = EdgeKind.DATAFLOW
    b.link("cam", "perception", df, True); b.link("lidar", "perception", df, True)
    b.link("radar", "fusion", df, True); b.link("gnss", "fusion", df, True); b.link("imu", "fusion", df, True)
    b.link("perception", "fusion", df, True); b.link("perception", "prediction", df, True)
    b.link("fusion", "planning", df, True); b.link("prediction", "planning", df, True)
    b.link("planning", "control", df, True)
    for a in ("steer", "throttle", "brake"):
        b.link("control", a, df, True); b.link(a, "vehicle_body", EdgeKind.PHYSICAL)
    b.link("vehicle_body", "imu", df, True); b.link("vehicle_body", "gnss", df, True)
    return b.build()

def franka_joint_graph(n_joints: int = 7) -> StructuralGraph:
    """Joint-level Franka: the actuated joints in a kinematic chain, each a failable node observing pos/vel/torque.
    The clean per-joint representation for the actuator-fault experiments and the momentum-observer comparison."""
    b = _Builder("franka_joint")
    for j in range(1, n_joints + 1):
        b.add(f"J{j}", NodeType.ACTUATOR)
        if j > 1:
            b.link(f"J{j-1}", f"J{j}")
    return b.build()


def anymal_sim_graph() -> StructuralGraph:
    """ANYmal-C laid out in Isaac Lab's own joint order: base + IMU + 12 leg actuators + 4 foot contacts.
    Node i>=2 maps to sim joint i-2. Unlike the manipulator chain, a hip fault here propagates into DIFFERENT
    node types (base tilt, foot contact), which is what makes the propagation operator well conditioned."""
    LEGS = ("LF", "LH", "RF", "RH")
    b = _Builder("anymal_sim")
    b.add("base", NodeType.BODY)
    b.add("imu", NodeType.SENSOR); b.link("base", "imu")
    for jt in ("HAA", "HFE", "KFE"):
        for leg in LEGS:
            b.add(f"{leg}_{jt}", NodeType.ACTUATOR)
    for leg in LEGS:
        b.link("base", f"{leg}_HAA"); b.link(f"{leg}_HAA", f"{leg}_HFE"); b.link(f"{leg}_HFE", f"{leg}_KFE")
    for leg in LEGS:
        b.add(f"{leg}_FOOT", NodeType.SENSOR); b.link(f"{leg}_KFE", f"{leg}_FOOT")
    return b.build()

def franka_sensor_graph(n_joints: int = 7) -> StructuralGraph:
    """Franka with the SENSING side split from the ACTUATION side, so a sensor is a candidate source in its own
    right rather than a channel of the joint it measures.

    Node order (used by scripts/isaac/gen_franka_sensor.py to fill the signal array): J1..Jn then E1..En.

        Jj  ACTUATOR  (applied_torque, pos_target - measured_pos, measured_vel * applied_torque)
        Ej  SENSOR    (measured_pos, measured_vel, measured_acc)          -- the joint encoder

    The split is what makes a sensor fault localizable at all. Keeping (pos, vel, torque) on one node per joint,
    as `franka_joint_graph` does, means a corrupted encoder and a weak actuator move the SAME node's channels and
    no method can be asked to tell them apart -- the label would not be a function of the graph. Split, each
    carries the evidence that identifies it: an actuator fault fails to track (pos_error at Jj grows while the
    commanded torque saturates), an encoder fault makes the servo track a lie (measured_pos at Ej jumps, and the
    controller's response drives the joint the wrong way, so Jj becomes the encoder's VICTIM). That victim
    relation is the point: the loudest node under an encoder fault is very often the actuator, not the encoder."""
    b = _Builder("franka_sensor")
    for j in range(1, n_joints + 1):
        b.add(f"J{j}", NodeType.ACTUATOR)
        if j > 1:
            b.link(f"J{j-1}", f"J{j}")
    for j in range(1, n_joints + 1):
        b.add(f"E{j}", NodeType.SENSOR); b.link(f"J{j}", f"E{j}")
    return b.build()


def anymal_sensor_graph() -> StructuralGraph:
    """ANYmal-C in Isaac Lab's joint order with the graph's sensor nodes made INJECTABLE.

    Same 18 nodes and the same order as `anymal_sim_graph` (base, imu, 12 leg actuators, 4 feet), so labels and
    node ids carry over, but the channel assignment is fixed so each node reports what that component actually
    measures and nothing else (4 channels):

        0      base   BODY      (lin_vel_x, lin_vel_y, lin_vel_z, height)      -- state estimate
        1      imu    SENSOR    (proj_grav_x, proj_grav_y, ang_vel_x, ang_vel_y)
        2-13   joints ACTUATOR  (joint_pos, joint_vel, applied_torque, pos_target - joint_pos)
        14-17  feet   SENSOR    (|F|, contact, F_z, |F_xy|)

    `anymal_sim_graph` gave the base node (proj_grav_z, ang_vel_x, ang_vel_y) and the imu node
    (proj_grav_x, proj_grav_y, ang_vel_z) -- an arbitrary split of ONE physical IMU across two nodes, which makes
    an IMU fault unattributable: corrupting the imu node alone is physically incoherent (a biased gyro biases
    every axis it measures) and corrupting both breaks the single-source label. Here the imu node owns the IMU
    and the base node carries the state estimator's own output, which on a real quadruped is a distinct signal
    fused from leg kinematics. The foot nodes gain their contact flag and force components, which were left as
    zeros before -- a contact-sensor fault is invisible in the force norm alone once the foot is off the ground."""
    LEGS = ("LF", "LH", "RF", "RH")
    b = _Builder("anymal_sensor")
    b.add("base", NodeType.BODY)
    b.add("imu", NodeType.SENSOR); b.link("base", "imu")
    for jt in ("HAA", "HFE", "KFE"):
        for leg in LEGS:
            b.add(f"{leg}_{jt}", NodeType.ACTUATOR)
    for leg in LEGS:
        b.link("base", f"{leg}_HAA"); b.link(f"{leg}_HAA", f"{leg}_HFE"); b.link(f"{leg}_HFE", f"{leg}_KFE")
    for leg in LEGS:
        b.add(f"{leg}_FOOT", NodeType.SENSOR); b.link(f"{leg}_KFE", f"{leg}_FOOT")
    return b.build()


EMBODIMENTS = {"franka": franka_graph, "anymal": anymal_graph,
               "crazyflie": crazyflie_graph, "vehicle": vehicle_graph}

