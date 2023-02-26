from setuptools import setup, find_packages

setup(
    name="bingo",
    version="0.1",
    description="Optimal gate syntehsis of parameterized quantum circuits. Built with tensorflow and qutip in python.",
    author="Alec Eickbusch, Jacob Curtis, Volodymyr Sivak, Shantanu Jha",
    author_email="alec.eickbusch@yale.edu",
    url="https://github.com/alec-eickbusch/bingo/branches",
    packages=["bingo/gate_sets", "bingo/optimizer"],
    install_requires=["qutip", "tensorflow", "h5py"],
)
