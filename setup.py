import os
from glob import glob

from setuptools import find_packages, setup

package_name = "hound_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["cuda/*"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "dora"), glob("dora/*")),
    ],
    install_requires=["setuptools", "numpy", "pyyaml", "pyarrow"],
    zip_safe=False,
    maintainer="hound",
    maintainer_email="hound@todo.todo",
    description="Dora manager/planner/controller nav adapter for HOUND (IGHA* + mppi).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "nav_node = hound_nav.nav_node:main",
            "nav_manager = hound_nav.manager_dora_node:main",
            "nav_planner = hound_nav.planner_dora_node:main",
            "nav_controller = hound_nav.controller_dora_node:main",
            "jit_build_cuda = hound_nav.jit_build:main",
        ],
    },
)
