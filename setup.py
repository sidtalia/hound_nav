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
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=False,
    maintainer="hound",
    maintainer_email="hound@todo.todo",
    description="IGHA* + UW_mppi ROS navigation adapter for HOUND.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "nav_node = hound_nav.nav_node:main",
            "nav_ipc_latency_probe = hound_nav.nav_ipc_latency_probe:main",
        ],
    },
)
