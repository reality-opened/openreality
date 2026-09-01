from setuptools import setup, find_packages

setup(
    name='openreality-core',
    version='2.2.2',
    description='OpenReality core — dense feed-forward monocular SLAM with VGGT-Omega',
    author='David Zhang',
    packages=find_packages(include=[
        'vggt_slam', 'vggt_slam.*',
    ]),
    extras_require={
        # Optional GPU photometric refinement for the 3DGS splat export
        # (vggt_slam.splat_export.refine_with_gsplat). Guarded by a lazy import;
        # the deterministic CPU baseline export needs none of this.
        # Install with: pip install -e ".[splat]"
        "splat": ["gsplat"],
    },
)

