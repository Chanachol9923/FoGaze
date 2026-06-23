import os
from glob import glob
from setuptools import find_packages, setup

package_name = "fogaze_rviz"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.rviz")),
        (os.path.join("lib", package_name), glob("scripts/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "fogaze_node = fogaze_rviz.fogaze_node:main",
        ],
    },
)
