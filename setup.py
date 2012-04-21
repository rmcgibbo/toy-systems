from distutils.core import setup
import glob

setup(name='toy_systems',
      version='0.1',
      description='toy_systems',
      author='Robert McGibbon',
      packages=['toy_systems'],#,'toy_systems.scripts'],
      package_dir={"toy_systems":'lib'},#,"toy_systems.scripts":'scripts'},
      #scripts=glob.glob('scripts/*'),
)
