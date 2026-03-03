"""Universal MuJoCo XML clone utilities.

Clone a single-asset XML to a new path, or generate a twin model (e.g. dataset + sim
bodies/actuators) from a single-body template so you don't maintain two files by hand.
"""

from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET


def clone_xml(
    src_path: str | Path,
    dst_path: str | Path,
    model_name: str | None = None,
) -> Path:
    """Copy a MuJoCo XML file to a new path, optionally changing the root model name.

    Does not resolve includes or copy external assets; use for self-contained XMLs
    or when asset paths are already absolute/relative to the new location.

    Args:
        src_path: Path to source XML.
        dst_path: Path to write the clone.
        model_name: If set, set the root <mujoco model="..."> to this name.

    Returns:
        Resolved dst_path as Path.
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        raise FileNotFoundError(f"Source XML not found: {src}")

    if model_name is None:
        shutil.copy2(src, dst)
        return dst.resolve()

    tree = ET.parse(src)
    root = tree.getroot()
    root.set("model", model_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_tree(tree, dst)
    return dst.resolve()


def make_twin_xml(
    src_path: str | Path,
    dst_path: str | Path,
    body_suffixes: Sequence[str] = ("_dataset", "_sim"),
    clone_positions: Sequence[str] | None = None,
    clone_geom_rgba: Sequence[str | None] | None = None,
    twin_model_name: str | None = None,
    actuator_name_for_suffix: str | None = None,
) -> Path:
    """Generate a twin MuJoCo XML from a single-body template.

    Finds the first movable body under <worldbody> (a direct child that contains
    a <joint>), duplicates it once per suffix, renames body/joint/geom/actuator
    so you get N independent copies (e.g. one for dataset replay, one for sim).
    Shared elements (default, visual, asset, option, floor, lights) are kept once.

    Args:
        src_path: Path to source single-body XML.
        dst_path: Path to write the twin XML.
        body_suffixes: Suffixes for each clone, e.g. ("_dataset", "_sim").
        clone_positions: Optional "x y z" position for each clone body (same order
            as body_suffixes). If shorter than body_suffixes, later clones keep
            the original body pos.
        clone_geom_rgba: Optional "r g b a" override for all <geom> elements in
            each clone body (e.g. (None, "0.1 1 1 0.8")). Use None for clones
            that should keep original colors.
        twin_model_name: Root model name in the output XML. Default: derived from
            source model name (e.g. roki_single_motor -> roki_twin_motor).
        actuator_name_for_suffix: Optional pattern for actuator names: "motor"
            yields motor_dataset, motor_sim. Default "motor".

    Returns:
        Resolved dst_path as Path.
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        raise FileNotFoundError(f"Source XML not found: {src}")

    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML has no <worldbody>")

    # Find first direct child body that contains a joint (the "robot" body)
    template_body = None
    for body in worldbody:
        if body.tag != "body":
            continue
        if body.find(".//joint") is not None:
            template_body = body
            break
    if template_body is None:
        raise ValueError("No direct child <body> with a <joint> found under <worldbody>")

    # Collect original names we need to rewrite in actuators
    joint_elem = template_body.find("joint")
    geom_elem = template_body.find("geom")
    original_body_name = template_body.get("name", "body")
    original_joint_name = joint_elem.get("name", "hinge") if joint_elem is not None else "hinge"
    actuator_name_base = actuator_name_for_suffix or "motor"

    # Set twin model name
    if twin_model_name is None:
        old_model = root.get("model", "model")
        twin_model_name = re.sub(r"single|one|1", "twin", old_model, flags=re.IGNORECASE)
        if twin_model_name == old_model:
            twin_model_name = f"{old_model}_twin"
    root.set("model", twin_model_name)

    # Remove the single template body; we'll add N clones
    worldbody.remove(template_body)

    positions = list(clone_positions) if clone_positions else []
    rgba_list = list(clone_geom_rgba) if clone_geom_rgba else []

    for i, suffix in enumerate(body_suffixes):
        clone = copy.deepcopy(template_body)
        _add_suffix_to_subtree(clone, suffix)
        new_body_name = (original_body_name + suffix) if original_body_name else f"body{suffix}"
        clone.set("name", new_body_name)
        if i < len(positions):
            clone.set("pos", positions[i])
        worldbody.append(clone)

        # Optional: set rgba on all geoms in this clone
        if i < len(rgba_list) and rgba_list[i] is not None:
            for geom in clone.iter("geom"):
                geom.set("rgba", str(rgba_list[i]))

    # Duplicate actuators: find actuator block and replace single-joint motors with N
    actuator = root.find("actuator")
    if actuator is not None and original_joint_name:
        # Find actuators that reference the original joint
        to_remove = []
        new_actuators = []
        for child in actuator:
            joint_ref = child.get("joint")
            if joint_ref != original_joint_name:
                continue
            to_remove.append(child)
            for suffix in body_suffixes:
                new_act = copy.deepcopy(child)
                new_act.set("joint", original_joint_name + suffix)
                new_act.set("name", actuator_name_base + suffix)
                new_actuators.append(new_act)
        for el in to_remove:
            actuator.remove(el)
        for el in new_actuators:
            actuator.append(el)

    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_tree(tree, dst)
    return dst.resolve()


def _add_suffix_to_subtree(node: ET.Element, suffix: str) -> None:
    """Recursively add suffix to 'name' and 'joint' attributes that look like IDs."""
    name = node.get("name")
    if name is not None and not name.endswith(suffix):
        node.set("name", name + suffix)
    joint = node.get("joint")
    if joint is not None and not joint.endswith(suffix):
        node.set("joint", joint + suffix)
    for child in node:
        _add_suffix_to_subtree(child, suffix)


def _write_tree(tree: ET.ElementTree, path: Path) -> None:
    """Write element tree to path with default XML declaration and MuJoCo-friendly formatting."""
    tree.write(
        path,
        encoding="unicode",
        default_namespace=None,
        method="xml",
    )
    # Ensure file ends with newline
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        path.write_text(text + "\n", encoding="utf-8")
