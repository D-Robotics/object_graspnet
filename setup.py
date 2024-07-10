from setuptools import find_packages, setup

package_name = 'object_graspnet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name + '/config', 
            ['config/' + 'grasp_horizon_2500.onnx']),
        ('lib/' + package_name + '/config', 
            ['config/' + 'depth.png']),
        ('lib/' + package_name + '/config', 
            ['config/' + 'cam_info.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zixi01.chen',
    maintainer_email='zixi01.chen@horizon.cc',
    description='TogetheROS object graspnet',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_graspnet_node = object_graspnet.object_graspnet_node:main',
        ],
    },
)
