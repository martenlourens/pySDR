from setuptools import setup, find_packages

setup(
    name='pySDR',
    version='0.1.0',
    author="Marten Lourens",
    packages=find_packages(include=['pySDR', 'pySDR.*']),
    install_requires=[
        'numpy>=1.20.3',
        'matplotlib>=3.4.3',
        'scikit-learn>=1.1.0',
        'umap-learn>=0.5.3'
    ],
    package_data={'pySDR' : ['*.so', '*.dll']}
    )