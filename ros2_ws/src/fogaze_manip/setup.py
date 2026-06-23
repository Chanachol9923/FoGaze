import os
from glob import glob
from setuptools import find_packages, setup

package_name = "fogaze_manip"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="FoGaze manipulation stack (graspability + arm pick).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pickup_planner = fogaze_manip.pickup_planner:main",
            "mock_arm_executor = fogaze_manip.mock_arm_executor:main",
            "moveit_pick_executor = fogaze_manip.moveit_pick_executor:main",
        ],
    },
)
