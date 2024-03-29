from setuptools import setup, find_packages
import versioneer
setup(name='deeprl',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author='Jason Rudy',
      author_email='',
      url='https://github.com/jcrudy/drlnd_p2',
      packages=find_packages(),
      requires=['six', 'numpy', 'pytorch', 'toolz', 'multipledispatch',
                'infinity', 'gym', 'mlagents']
     )