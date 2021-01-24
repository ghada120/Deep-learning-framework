from setuptools import setup, find_packages


setup(
    name='PyloXyloto',
    version='1.0',
    description='PyloXyloto is a simple Deep learning framework built from scratch with python that supports the main '
                'functionalities needed for a deep learning project',
    py_modules=["activations, losses, layers, data, metrics, utils, visualization"],
    package_dir={'': 'src'},
    author='Ahmed Mohamed, Ghada Ahmed : AinShams University',
    keywords=['DeepLearning', 'FrameWork', 'NeuralNetworks', 'TensorFlow', 'Pytorch', 'Python'],
    url='https://github.com/ghada120/Deep-learning-framework',
    download_url='https://pypi.org/project/PyloXyloto/',
)


