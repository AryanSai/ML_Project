from setuptools import setup, find_packages

HYPEN_E_DASH = '-e .'

def get_requirements(file_path):
    """
    This function reads a requirements file and returns a list of packages.
    It removes any comments and empty lines.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        if HYPEN_E_DASH in requirements:
            requirements.remove(HYPEN_E_DASH)
    
    # Clean up the requirements list
    requirements = [line.strip() for line in requirements if line.strip() and not line.startswith('#')]
    
    return requirements

setup(
    name='ml_project',
    version='0.0.1',
    author='Aryan Sai',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)