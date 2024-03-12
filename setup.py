from setuptools import setup


PACKAGES = [
        'rotation_forest',
        'rotation_forest.tests'
]

def setup_package():
    setup(
        name="rotation_forest",
        version='1.0',
        description='Sklearn style implementation of the Rotation Forest Algorithm',
        long_description='''
        Implementation of the Algorithm: J. J. Rodriguez, L. I. Kuncheva and C. J. Alonso, "Rotation Forest: A New Classifier Ensemble Method," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 28, no. 10, pp. 1619-1630, Oct. 2006.
doi: 10.1109/TPAMI.2006.211''',
        author='Joshua D. Loyal, Abhisek Maiti',
        author_email='mail2abhisek.maiti@gmail.com',
        maintainer='Abhisek Maiti',
        maintainer_email='mail2abhisek.maiti@gmail.com',
        url='https://github.com/digital-idiot/RotationForest',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
