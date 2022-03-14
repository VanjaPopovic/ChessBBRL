from setuptools import setup, find_packages

setup(name="behaviour_gym",
    version="0.1.0",
    auther="Lewis Boyd",
    author_email="l.boyd@strath.ac.uk",
    description="Robotic Gym environments implemented in PyBullet",
    packages=find_packages(include=[
        "behaviour_gym",
        "behaviour_gym.*"
    ]),
    install_requires=[
        "pybullet",
        "gym",
        "numpy",
        "opencv-python"
    ]
)
