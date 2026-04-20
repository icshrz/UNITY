from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths


def main():
    curr_dir = Path(__file__).absolute().parent
    setup(
        name="unignn",
        ext_modules=[
            CppExtension(
                name="unignn",
                sources=[
                    # "lib/concurrentcy.cpp",
                    # "lib/mapping.cpp",
                    # "lib/parallel.cpp",
                    # "lib/sampler.cpp",
                    "lib/block_manager.cpp",
                    "lib/table.cpp",
                ],
                include_dirs=[curr_dir / "include/"],
                extra_compile_args=["-std=c++14", "-fopenmp"],
                extra_link_args=["-fopenmp"],
            )
        ],
        cmdclass={
            "build_ext": BuildExtension
        }
    )


if __name__ == "__main__":
    main()
