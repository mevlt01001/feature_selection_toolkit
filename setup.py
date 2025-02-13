from setuptools import setup, find_packages

setup(
    name='feature_selection_toolkit',
    version='1.1.0',
    author='Mevlüt Başaran',
    author_email='mevlutbasaran01@gmail.com',
    description='A comprehensive toolkit for performing various feature selection techniques in machine learning.',
    license=open('LICENSE').read(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mevlt01001/feature_selection_toolkit',
    project_urls={
        'GitHub': 'https://github.com/mevlt01001/feature_selection_toolkit',
        'Bug Tracker': 'https://github.com/mevlt01001/feature_selection_toolkit/issues',
        'LinkedIn': 'https://www.linkedin.com/in/mevl%C3%BCt-ba%C5%9Faran-b46888251/',
        'Kaggle': 'https://www.kaggle.com/mevltbaaran',
    },
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
        'pandas',
        'scikit-learn',
        'tqdm',
        'xgboost',
        'statsmodels',
        'joblib',
        'matplotlib',
        'numpy',
    ],
    test_suite='feature_selection_test',  # Ensure this is the correct path to your test suite
    tests_require=['pytest'],
)
