from setuptools import setup


PACKAGES = [
        'rotation_forest',
        'rotation_forest.tests'
]

def setup_package():
    setup(
        name="rotation_forest",
        version='0.2',
        description='Sklearn style implementation of the Rotation Forest Algorithm',
        author='Joshua D. Loyal, Abhisek Maiti',
        author_email='mail2abhisek.maiti@gmail.com'
        url='https://github.com/digital-idiot/RotationForest',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
