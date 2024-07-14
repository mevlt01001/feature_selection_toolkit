from setuptools import setup, find_packages

setup(
    name='feature_selection_toolkit',
    version='1.0.0',
    author='Mevlüt Başaran',
    author_email='mevlutbasaran01@gmail.com',
    description='A comprehensive toolkit for performing various feature selection techniques in machine learning.',
    license=open('LICENSE').read(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mevlut01001/feature_selection_toolkit',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pandas >= 2.2.2',
        'scikit-learn >= 1.4.2',
        'tqdm >= 4.66.4',
        'xgboost >= 2.1.0',
        'statsmodels >= 0.14.2',
        'joblib >= 1.4.2',
        'matplotlib >= 3.8.4',
        'numpy >= 1.26.4',
    ],
    test_suite='feature_selection_test',
    tests_require=['pytest'],
)
