'''
Building guideline:

In the current directory, running `python setup.py install` If the
operation is not authorized, try `python setup.py install --user`.

'''

from setuptools import setup, find_packages

setup(
    name='uf',
    version='beta v2.9.2',
    description='Unified framework for NLP tasks.',
    url='https://github.com/geyingli/unif',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Geying Li',
    author_email='geyingli@tencent.com',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    extras_require={
        'cpu': ['tensorflow>=1.11.0'],
        'gpu': ['tensorflow-gpu>=1.11.0'],
    },
    python_requires=">=3.6.0",
    classifiers=[
        'Operating System :: OS Independent',
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=(
        'bert xlnet electra nlp tensorflow classification generation '
        'question-answering machine-reading-comprehension '
        'translation sequence-labeling'),
)
