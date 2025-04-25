from setuptools import setup, find_packages

setup(
    name="flappy-ai",  # Name of your package
    version="1.0.0",     # Version of your package
    packages=find_packages(),  # Automatically find all packages inside your package directory
    include_package_data=True,  # Includes files specified in MANIFEST.in (if any)
    install_requires=[    # List any external packages your package needs
        # "somepackage>=1.0",  # Example: replace with actual dependencies if any
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',  # Minimum Python version
)