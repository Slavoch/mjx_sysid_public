from setuptools import setup, find_packages

setup(
    name="sysidmjx",  # Your package name
    version="0.1.0",  # Version of the package
    author="Vyacheslav Kovalev",  # Your name
    author_email="ssaavvaa222@gmail.com",  # Your email
    description="TODO",  # Short description
    long_description=open("README.md").read(),  # Long description from your README
    long_description_content_type="text/markdown",  # Type of README (markdown in this case)
    url="https://wavegit.mipt.ru/Slavoch/mjx_sysid",  # URL to the project (optional)
    # license="MIT",  # License type (MIT, Apache, etc.)
    packages=find_packages(),  # Automatically find your package and subpackages
    install_requires=[
        # List your package dependencies here
        # 'numpy',
        # 'pandas',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Python version requirement
)
