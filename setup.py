from setuptools import setup

with open("requirements.txt", "r") as req:
    requires = req.read().split("\n")


setup(name="gppath",
      version="0.1",
      description="Informative path planning with output-weighted Bayesian optimization",
      author="Antoine Blanchard",
      author_email="ablancha@mit.edu",
      install_requires=requires,
      packages=setuptools.find_packages(),
      include_package_data=True,
      license="MIT"
    )
