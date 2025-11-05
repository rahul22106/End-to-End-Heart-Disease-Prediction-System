from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# edit the below metadata as per your project requirements
REPO_NAME = "End-to-End-Heart-Disease-Prediction-System"
AUTHOR_USER_NAME = "Rahul Kumar Mishra"
SCR_NAME = "Heart_Disease_Prediction"
LIST_OF_REQUIREMENTS = []

setup(
    name=SCR_NAME,
    version="0.0.1",
    author="Rahul Kumar Mishra",
    description="An end-to-end heart Disease Prediction system project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rahul22106/End-to-End-Heart-Disease-Prediction-System.git",
    author_email="rahulkrmishra2004@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS,
)
