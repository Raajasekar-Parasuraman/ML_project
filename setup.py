from setuptools import find_packages,setup

def get_requirement(file_path):
    '''
    This function returns the list of Library requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()

    requirements=[i.replace('\n','') for i in requirements]
    return requirements



setup(
name='MLproject',
version='0.0.1',
author='Raaja',
author_email='raajavprs36@gmail.com',
packages=find_packages(),
install_requires=get_requirement('requirements.txt')

)