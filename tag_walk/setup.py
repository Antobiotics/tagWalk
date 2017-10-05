from setuptools import setup

try:
    import multiprocessing
except ImportError:
    pass

setup(
    name="fachung",
    version='1.0.0',
    description="TF Recommendations and Embeddings",
    author_email="greg@dice.fm",
    url="https://github.com/antobiotics/fachung",
    platforms="Posix; MacOS X; Windows",
    entry_points={
        "console_scripts": ['fachung = fachung.main:main']
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
    ],
    packages=[
        "fachung",
        "fachung.commands",
        "fachung.commands.builders",
        "fachung.preparation",
        "fachung.datasets",
        "fachung.models"
    ],
    setup_requires=['numpy', 'h5py'],
    install_requires=[
        "psycopg2",
        "sklearn",
        "click",
        "coloredlogs",
        "executor",
        "numpy",
        "keras",
        "pandas",
        "luigi",
        "scipy",
        "nibabel",
        "tqdm",
        "executor"
    ]
)
