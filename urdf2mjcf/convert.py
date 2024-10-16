"""Uses Mujoco to convert from URDF to MJCF files."""

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from urdf2mjcf.utils import iter_meshes, save_xml


def add_compiler(root: ET.Element) -> None:
    if isinstance(element := root.find("compiler"), ET.Element):
        root.remove(element)
    element = ET.Element(
        "compiler",
        attrib={
            "angle": "radian",
            "meshdir": "meshes",
            "eulerseq": "zyx",
            "autolimits": "true",
        },
    )
    root.insert(0, element)


def add_default(root: ET.Element) -> None:
    if isinstance(element := root.find("default"), ET.Element):
        root.remove(element)
    default = ET.Element("default")

    # Adds default joint options.
    ET.SubElement(
        default,
        "joint",
        attrib={
            "limited": "true",
            "damping": "0.01",
            "armature": "0.01",
            "frictionloss": "0.01",
            "solref": "0.001 2",
        },
    )

    # Adds default geom options.
    ET.SubElement(
        default,
        "geom",
        attrib={
            "condim": "4",
            "contype": "1",
            "conaffinity": "15",
            "friction": "0.9 0.2 0.2",
            "solref": "0.001 2",
        },
    )

    # Adds default motor options.
    if element := default.find("motor") is not None:
        default.remove(element)
    element = ET.SubElement(
        default,
        "motor",
        attrib={"ctrllimited": "true"},
    )

    # Adds default equality options.
    if element := default.find("equality") is not None:
        default.remove(element)
    element = ET.SubElement(
        default,
        "equality",
        attrib={"solref": "0.001 2"},
    )

    # Adds default visualgeom options.
    if element := default.find("default") is not None:
        default.remove(element)
    default_element = ET.SubElement(
        default,
        "default",
        attrib={"class": "visualgeom"},
    )
    element = ET.SubElement(
        default_element,
        "geom",
        attrib={"material": "visualgeom", "condim": "1", "contype": "0", "conaffinity": "0"},
    )

    root.insert(0, default)


def add_option(root: ET.Element) -> None:
    element = ET.SubElement(
        root,
        "option",
        attrib={
            "iterations": "50",
            "timestep": "0.001",
            "solver": "PGS",
            "gravity": "0 0 -9.81",
        },
    )
    root.insert(0, element)


def convert_urdf_to_mjcf(
    urdf_path: str | Path,
    mjcf_path: str | Path | None = None,
    no_collision_mesh: bool = False,
) -> None:
    """Convert a URDF file to an MJCF file.

    Args:
        urdf_path: The path to the URDF file.
        mjcf_path: The path to the MJCF file. If not provided, use the URDF
            path with the extension replaced with ".mjcf".
        no_collision_mesh: Do not include collision meshes.
    """
    urdf_path = Path(urdf_path)
    mjcf_path = Path(mjcf_path) if mjcf_path is not None else urdf_path.with_suffix(".mjcf")
    if not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Copy URDF file to temp directory
        temp_urdf_path = temp_dir_path / urdf_path.name
        temp_urdf_path.symlink_to(urdf_path)

        # Copy mesh files to temp directory
        for (_, visual_mesh_path), (_, collision_mesh_path) in iter_meshes(urdf_path):
            for mesh_path in list({visual_mesh_path, collision_mesh_path}):
                if mesh_path is not None:
                    temp_mesh_path = temp_dir_path / mesh_path.name
                    try:
                        temp_mesh_path.symlink_to(mesh_path)
                    except FileExistsError:
                        pass

        # Load the URDF file with Mujoco and save it as an MJCF file in the temp directory
        temp_mjcf_path = temp_dir_path / mjcf_path.name
        model = mujoco.MjModel.from_xml_path(temp_urdf_path.as_posix())
        mujoco.mj_saveLastXML(temp_mjcf_path.as_posix(), model)

        # Read the MJCF file and update the paths to the meshes
        mjcf_tree = ET.parse(temp_mjcf_path)

        for asset in mjcf_tree.iter("asset"):
            for mesh in asset.iter("mesh"):
                mesh_name = Path(mesh.attrib["file"]).name
                original_mesh_path = next(p for p in urdf_path.parent.glob("**/*") if p.name == mesh_name)
                mesh.attrib["file"] = original_mesh_path.relative_to(urdf_path.parent).as_posix()

        # Turn off internal collisions
        if not no_collision_mesh:
            root = mjcf_tree.getroot()
            for element in root:
                if element.tag == "geom":
                    element.attrib["contype"] = str(1)
                    element.attrib["conaffinity"] = str(0)

        # Manually set additional options.
        add_default(root)
        add_compiler(root)

        # Write the updated MJCF file to the original destination
        save_xml(mjcf_path, mjcf_tree)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Convert a URDF file to an MJCF file.")
    parser.add_argument("urdf_path", type=str, help="The path to the URDF file.")
    parser.add_argument("--no-collision-mesh", action="store_true", help="Do not include collision meshes.")
    parser.add_argument("--output", type=str, help="The path to the output MJCF file.")
    args = parser.parse_args()

    convert_urdf_to_mjcf(
        urdf_path=args.urdf_path,
        mjcf_path=args.output,
        no_collision_mesh=args.no_collision_mesh,
    )


if __name__ == "__main__":
    cli()
