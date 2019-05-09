# Copyright 2015-2018 The MathWorks, Inc.

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

if __name__ == '__main__':

    setup(
        name="matlabruntimeforpython",
        version="R2019a",
        description='A module to call MATLAB from Python',
        author='MathWorks',
        url='https://www.mathworks.com/',
        platforms=['Linux', 'Windows', 'MacOS'],
        packages=[
            'TIMERS_ALL'
        ],
        package_data={'TIMERS_ALL': ['*.ctf']},
        # Executes the custom code above in order to delete the build area.
        cmdclass={'install': InstallRuntime}
    )


