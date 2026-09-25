"""Build a StructuralGraph straight from a URDF, so the dataset's graphs are reproducible from the same robot
description the simulator loads. Links become link nodes, non-fixed joints become actuator nodes wired to their
parent and child links; optionally a torque sensor is attached per actuated joint and an IMU to a chosen link."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from ..core.graph import StructuralGraph, Node, Edge, NodeType, EdgeKind

def graph_from_urdf(source: str, name: str = "", add_joint_sensors=True, imu_link: str | None = None) -> StructuralGraph:
    xml = Path(source).read_text() if Path(source).exists() else source
    root = ET.fromstring(xml)
    name = name or root.get("name", "robot")
    link_names = [l.get("name") for l in root.findall("link")]
    idx, nodes, edges = {}, [], []
    def add(nm, tp):
        idx[nm] = len(nodes); nodes.append(Node(len(nodes), nm, tp)); return idx[nm]
    base = link_names[0] if link_names else "base"
    for i, ln in enumerate(link_names):
        add(ln, NodeType.BODY if ln == base else NodeType.LINK)
    for j in root.findall("joint"):
        jn, jt = j.get("name"), j.get("type", "fixed")
        parent = j.find("parent").get("link") if j.find("parent") is not None else None
        child = j.find("child").get("link") if j.find("child") is not None else None
        if parent not in idx or child not in idx:
            continue
        if jt in ("fixed",):
            edges.append(Edge(idx[parent], idx[child], EdgeKind.PHYSICAL))
            continue
        a = add(jn, NodeType.ACTUATOR)
        edges += [Edge(idx[parent], a, EdgeKind.PHYSICAL), Edge(a, idx[child], EdgeKind.PHYSICAL)]
        if add_joint_sensors:
            s = add(f"{jn}_torque", NodeType.SENSOR); edges.append(Edge(a, s, EdgeKind.PHYSICAL))
    if imu_link and imu_link in idx:
        s = add("imu", NodeType.SENSOR); edges.append(Edge(idx[imu_link], s, EdgeKind.PHYSICAL))
    return StructuralGraph(nodes, edges, name)
