from setuptools import find_packages, setup

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="leads-jarvis",
    version="0.0.1-alpha.2",
    python_requires=">=3.12",
    author="ProjectNeura",
    author_email="central@projectneura.org",
    description="Jarvis Extension for LEADS",
    license="Apache License 2.0",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ProjectNeura/LEADS-Jarvis",
    packages=find_packages(),
    package_data={
        "leads_jarvis": ["checkpoints/*"]
    },
    include_package_data=True,
    install_requires=["leads>=0.9.1", "torch", "torchvision", "timm", "segment-anything", "requests", "rich",
                      "ultralytics"]
)
