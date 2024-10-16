"""Uses Mujoco to convert from URDF to MJCF files."""

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from urdf2mjcf.utils import iter_meshes, save_xml


def add_compiler(root: ET.Element) -> None:
    element = ET.Element(
        "compiler",
        attrib={
            "angle": "radian",
            "meshdir": "meshes",
            "eulerseq": "zyx",
            "autolimits": "true",
        },
    )

    if isinstance(existing_element := root.find("compiler"), ET.Element):
        root.remove(existing_element)
    root.insert(0, element)


def add_default(root: ET.Element) -> None:
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
    ET.SubElement(
        default,
        "motor",
        attrib={"ctrllimited": "true"},
    )

    # Adds default equality options.
    ET.SubElement(
        default,
        "equality",
        attrib={"solref": "0.001 2"},
    )

    # Adds default visualgeom options.
    default_element = ET.SubElement(
        default,
        "default",
        attrib={"class": "visualgeom"},
    )
    ET.SubElement(
        default_element,
        "geom",
        attrib={"material": "visualgeom", "condim": "1", "contype": "0", "conaffinity": "0"},
    )

    if isinstance(existing_element := root.find("default"), ET.Element):
        root.remove(existing_element)
    root.insert(0, default)


def add_option(root: ET.Element) -> None:
    element = ET.Element(
        "option",
        attrib={
            "iterations": "50",
            "timestep": "0.001",
            "solver": "PGS",
            "gravity": "0 0 -9.81",
        },
    )

    if isinstance(existing_element := root.find("option"), ET.Element):
        root.remove(existing_element)
    root.insert(0, element)


def add_assets(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add textures and materials
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "name": "texplane",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".0 .0 .0",
            "rgb2": ".8 .8 .8",
            "width": "100",
            "height": "108",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "matplane",
            "reflectance": "0.",
            "texture": "texplane",
            "texrepeat": "1 1",
            "texuniform": "true",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "visualgeom",
            "rgba": "0.5 0.9 0.2 1",
        },
    )


def add_worldbody_elements(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Add ground plane
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            attrib={
                "name": "ground",
                "type": "plane",
                "pos": "0.001 0 0",
                "quat": "1 0 0 0",
                "material": "matplane",
                "condim": "1",
                "conaffinity": "15",
                "size": "0 0 1",
            },
        ),
    )

    # Add lights
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.6 0.6 0.6",
                "specular": "0.2 0.2 0.2",
                "pos": "0 0 4",
                "dir": "0 0 -1",
            },
        ),
    )
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.4 0.4 0.4",
                "specular": "0.1 0.1 0.1",
                "pos": "0 0 5.0",
                "dir": "0 0 -1",
                "castshadow": "false",
            },
        ),
    )


def add_root_body(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Create a root body
    root_body = ET.Element(
        "body",
        attrib={
            "name": "root",
            "pos": "0 0 0",
            "quat": "0.0 0.0 0 1",
        },
    )

    # Add a freejoint
    ET.SubElement(
        root_body,
        "freejoint",
        attrib={"name": "root"},
    )

    # Add cameras
    ET.SubElement(
        root_body,
        "camera",
        attrib={
            "name": "front",
            "pos": "0 -3 1",
            "xyaxes": "1 0 0 0 1 2",
            "mode": "trackcom",
        },
    )
    ET.SubElement(
        root_body,
        "camera",
        attrib={
            "name": "side",
            "pos": "-2.893 -1.330 0.757",
            "xyaxes": "0.405 -0.914 0.000 0.419 0.186 0.889",
            "mode": "trackcom",
        },
    )

    # Add imu site
    ET.SubElement(
        root_body,
        "site",
        attrib={
            "name": "imu",
            "size": "0.01",
            "pos": "0 0 0",
        },
    )

    # Move existing bodies and geoms under root_body
    elements_to_move = list(worldbody)
    for elem in elements_to_move:
        if elem.tag in {"body", "geom"}:
            worldbody.remove(elem)
            root_body.append(elem)
    worldbody.append(root_body)


def add_actuators(root: ET.Element) -> None:
    actuator_element = ET.Element("actuator")

    # For each joint, add a motor actuator
    for joint in root.iter("joint"):
        joint_name = joint.attrib.get("name")
        if joint_name is None:
            continue
        # Get actuatorfrcrange if present
        actuatorfrcrange = joint.attrib.get("actuatorfrcrange")
        if actuatorfrcrange is not None:
            ctrlrange = actuatorfrcrange
        else:
            ctrlrange = "-1000.0 1000.0"

        ET.SubElement(
            actuator_element,
            "motor",
            attrib={
                "name": joint_name,
                "joint": joint_name,
                "ctrllimited": "true",
                "ctrlrange": ctrlrange,
                "gear": "1",
            },
        )

    if isinstance(existing_element := root.find("actuator"), ET.Element):
        root.remove(existing_element)
    root.append(actuator_element)


def add_sensors(root: ET.Element) -> None:
    sensor_element = ET.Element("sensor")

    # For each actuator, add sensors
    actuators = root.find("actuator")
    if actuators is not None:
        for actuator in actuators.iter("motor"):
            actuator_name = actuator.attrib.get("name")
            if actuator_name is None:
                continue

            # Add actuatorpos sensor
            ET.SubElement(
                sensor_element,
                "actuatorpos",
                attrib={
                    "name": f"{actuator_name}_p",
                    "actuator": actuator_name,
                },
            )

            # Add actuatorvel sensor
            ET.SubElement(
                sensor_element,
                "actuatorvel",
                attrib={
                    "name": f"{actuator_name}_v",
                    "actuator": actuator_name,
                },
            )

            # Add actuatorfrc sensor
            ET.SubElement(
                sensor_element,
                "actuatorfrc",
                attrib={
                    "name": f"{actuator_name}_f",
                    "actuator": actuator_name,
                    "noise": "0.001",
                },
            )

    # Add additional sensors
    # For example, framequat and gyro sensors
    # For this, we need to have a site named "imu"
    imu_site = None
    for site in root.iter("site"):
        if site.attrib.get("name") == "imu":
            imu_site = site
            break

    if imu_site is not None:
        # Add framequat sensor
        ET.SubElement(
            sensor_element,
            "framequat",
            attrib={
                "name": "orientation",
                "objtype": "site",
                "noise": "0.001",
                "objname": "imu",
            },
        )

        # Add gyro sensor
        ET.SubElement(
            sensor_element,
            "gyro",
            attrib={
                "name": "angular-velocity",
                "site": "imu",
                "noise": "0.005",
                "cutoff": "34.9",
            },
        )

    if isinstance(existing_element := root.find("sensor"), ET.Element):
        root.remove(existing_element)
    root.append(sensor_element)


def add_keyframes(root: ET.Element) -> None:
    keyframe_element = ET.Element("keyframe")

    # If you have specific keyframe data, you can add it here.
    # For now, we'll use a placeholder.
    ET.SubElement(
        keyframe_element,
        "key",
        attrib={
            "name": "default",
            "qpos": "0 0 0.63 1 0.0 0.0 0 -0.157 0.0394 0.0628 0.441 -0.258 -0.22 0.026 0.0314 0.441 -0.223",
        },
    )

    if isinstance(existing_element := root.find("keyframe"), ET.Element):
        root.remove(existing_element)
    root.append(keyframe_element)


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
        root = mjcf_tree.getroot()

        for asset in mjcf_tree.iter("asset"):
            for mesh in asset.iter("mesh"):
                mesh_name = Path(mesh.attrib["file"]).name
                # Update the file attribute to just the mesh name
                mesh.attrib["file"] = mesh_name

        # Turn off internal collisions
        if not no_collision_mesh:
            for geom in root.iter("geom"):
                geom.attrib["contype"] = str(1)
                geom.attrib["conaffinity"] = str(0)

        # Manually set additional options.
        add_default(root)
        add_compiler(root)
        add_option(root)
        add_assets(root)
        add_root_body(root)
        add_worldbody_elements(root)
        add_actuators(root)
        add_sensors(root)
        # add_keyframes(root)

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
