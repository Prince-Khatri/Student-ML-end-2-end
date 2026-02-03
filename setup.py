from setuptools import find_packages, setup
from typing import List

REQUIREMENTS_PATH = "requirements.txt"
HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List(str):
    """
        This is a function to return a list of requirements.txt from given path.
    """

    with open(file_path,'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    
        if HYPEN_E_DOT == "-e .":
            requirements.remove(HYPEN_E_DOT)
    
    return requirements
    
setup(
    name="mlproject",
    version="0.0.1",
    author="Prince",
    author_email = "princekhatri1013@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements(REQUIREMENTS_PATH)
)