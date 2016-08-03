from setuptools import setup


def setup_package():
    setup(
        name="rotation_forest",
        version='0.1.0',
        description='Sklearn style implementation of the Rotation Forest Algorithm',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/RotationForest',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn'],
    )


if __name__ == '__main__':
    setup_package()
