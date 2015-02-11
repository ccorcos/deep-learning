from setuptools import setup


setup(
    name='DL',
    version='0.1',
    description='Some deep learning tools.',
    long_description='Some deep learning tools.',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
    ],
    keywords='deep learning',
    url='https://github.com/ccorcos/',
    author='Chet Corcos',
    author_email='ccorcos@gmail',
    license='MIT',
    packages=['DL'],
    install_requires=[
        'numpy',
        'theano',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False
)
