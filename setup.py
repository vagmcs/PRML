import setuptools

with open("README.md", "r") as fh:
    long_text = fh.read()

setuptools.setup(
    name="PRML",
    version="0.1",
    author="Evangelos Michelioudakis",
    description="PRML notes and algorithms implemented in Python",
    long_description=long_text,
    long_description_content_type="text/markdown",
    license='GPL3',
    packages=setuptools.find_packages(),
    entry_points='''
        [console_scripts]
    ''',
    classifiers=[
        "Programming Language :: Python :: 3",
        "GNU General Public License v3': 'License :: OSI Approved :: GNU General Public License v3 (GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
