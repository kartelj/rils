from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='rils',
    version='0.4',
    description='Regression via Iterated Local Search for Symbolic Regression GECCO Competition -- 2023',
    long_description= long_description,
    long_description_content_type  = "text/markdown",
    author='Aleksandar Kartelj, Marko Đukanović',
    author_email='aleksandar.kartelj@gmail.com',
    url='https://github.com/kartelj/rils',
    packages = find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent"
    ],
    python_requires= ">=3.6",
    py_modules=["rils"],
    package_dir = {'rils':'rils'}, 
    install_requires=["numpy", "sympy", "scikit-learn","statsmodels"],
)