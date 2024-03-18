import setuptools

REQUIRED_PACKAGES = [
    "apache-beam[gcp]",
    "tensorflow==2.8.0",
    "gensim==3.6.0",
    "fsspec==0.8.4",
    "gcsfs==0.7.1",
    "numpy==1.20.0",
    "keras==2.8.0",
]

setuptools.setup(
    name="twitteremotion",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description="Cloud ML Twitter sentiment analysis with preprocessing",
)
