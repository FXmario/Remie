"""Mark wheels containing a native sandbox helper as platform-specific."""
from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class SandboxWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.bundled = bool(list(Path("remie/_vendor/bubblewrap").glob("linux-*/bwrap")))
        if self.bundled:
            self.root_is_pure = False

    def get_tag(self):
        python, abi, platform = super().get_tag()
        return ("py3", "none", platform) if self.bundled else (python, abi, platform)


setup(cmdclass={"bdist_wheel": SandboxWheel})
