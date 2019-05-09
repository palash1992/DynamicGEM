import os
from glob import glob
from setuptools import setup, find_packages
import subprocess
import imp

from distutils.core import setup
from distutils.command.clean import clean
from distutils.command.install import install


class InstallRuntime(install):
    # Calls the default run command, then deletes the build area
    # (equivalent to "setup clean --all").
    def run(self):
        install.run(self)
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()


DISTNAME = 'dynamicgem'
MAINTAINER = 'Palash Goyal, Sujit Rokka Chhetri'
MAINTAINER_EMAIL = 'palashgo@usc.edu, schhetri@uci.edu'
DESCRIPTION = 'dynamicGEM: A Python module for Dynamic Graph Embedding Methods'
LONG_DESCRIPTION = open('README.md').read()
URL = 'https://github.com/palash1992/DynamicGEM'
DOWNLOAD_URL = 'https://github.com/palash1992/DynamicGEM'
KEYWORDS = ['dynamic graph embedding', 'graph embedding', 'network analysis',
            'network embedding', 'data mining', 'machine learning']
LICENSE = 'BSD'
VERSION = '1.0.0'
ISRELEASED = True

INSTALL_REQUIRES = (
    'tensorflow==1.11.0',
    'Cython>=0.29',
    'h5py==2.8.0',
    'joblib>=0.12.5',
    'Keras>=2.2.4',
    'matplotlib==3.0.1',
    'networkx>=1.11',
    'numpy>=1.15.3',
    'pandas>=0.23.4',
    'scikit-learn>=0.20.0',
    'scipy>=1.1.0',
    'seaborn>=0.9.0',
    'six>=1.11.0',
    'sklearn>=0.0'
)


def get_package_data(topdir, excluded=set()):
    retval = []
    for dirname, subdirs, files in os.walk(os.path.join(DISTNAME, topdir)):
        for x in excluded:
            if x in subdirs:
                subdirs.remove(x)
        retval.append(os.path.join(dirname[len(DISTNAME) + 1:], '*.*'))
    return retval


def get_data_files(dest, source):
    retval = []
    for dirname, subdirs, files in os.walk(source):
        retval.append(
            (os.path.join(dest, dirname[len(source) + 1:]), glob(os.pathjoin(dirname, '*.*')))
        )
    return retval


# Return the git revision as a string
def git_version():
    """Return the git revision as a string.
    Copied from numpy setup.py
    """

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION


def write_version_py(filename='dynamicgem/version.py'):
    # Copied from numpy setup.py
    cnt = """
    # THIS FILE IS GENERATED FROM DynamicGEM SETUP.PY
    short_version = '%(version)s'
    version = '%(version)s'
    full_version = '%(full_version)s'
    git_revision = '%(git_revision)s'
    release = %(isrelease)s
    if not release:
        version = full_version
        short_version += ".dev"
    """
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('dynamicgem/version.py'):
        # must be a source distribution, use existing version file
        version = imp.load_source('DynamicGEM.version', 'dynamicgem/version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def setup_package():
    write_version_py()
    setup(
        name=DISTNAME,
        version=VERSION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        url=URL,
        download_url=DOWNLOAD_URL,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        packages=find_packages(),
        package_dir={DISTNAME: 'dynamicgem'},
        package_data={DISTNAME: get_package_data('datasets')},
        license=LICENSE,
        long_description=LONG_DESCRIPTION,
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence',
                     'Topic :: Scientific/Engineering :: Graph Analysis',
                     'Programming Language :: Python :: 3.5', ],
    )


def timer_setup():
    setup(
        name="matlabruntimeforpython",
        version="R2017a",
        description='A module to call MATLAB from Python',
        author='MathWorks',
        url='http://www.mathworks.com/',
        platforms=['Linux', 'Windows', 'MacOS'],
        packages=[ 'TIMERS_ALL' ],
        package_dir={'TIMERS_ALL': 'dynamicgem/TIMERS/TIMERS_ALL/for_redistribution_files_only/TIMERS_ALL'},
        package_data={'TIMERS_ALL': ['*.ctf']},
        # Executes the custom code above in order to delete the build area.
        cmdclass={'install': InstallRuntime}
    )


if __name__ == "__main__":
    setup_package()
    timer_setup()
