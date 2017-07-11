#try:
from setuptools import setup
#except:
    #from distutils.core import setup

try:
    import multiprocessing
except ImportError:
    pass

setup(
    name="tag_walk",
    version='1.0.0',
    description="TF Recommendations and Embeddings",
    author_email="greg@dice.fm",
    url="https://github.com/antobiotics/tag_walk",
    platforms="Posix; MacOS X; Windows",
    entry_points = {
        "console_scripts": ['tag_walk = tag_walk.main:main']
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
    ],
    packages=[
        "tag_walk",
        "tag_walk.commands",
        "tag_walk.commands.builders"
    ],
    install_requires=[
        "click",
        "executor",
        "numpy",
        "tensorflow",
        "keras",
        "pandas",
        "scipy"
    ],
    dependency_links=[
        'git+ssh://git@github.com/spotify/luigi.git',
    ]
)
