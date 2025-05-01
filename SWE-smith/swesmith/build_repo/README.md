# Automatic Repo Image Construction

To create an image for a repository...
1. [`test_install.py`]: Test out whether a repo can be installed successfully at a particular commit.
2. [`constants.py`]: Add an entry for the repo, containing:
```json
{
    "<commit>": {
        "install": "<install_command>",
        "packages": "custom_environment",
        "python": "<python_version>",
        "test_cmd": "<test_command>",
    }
}
```
3. [`create_images.py`]: Run this to make sure the repo image can be built successfully.

> ⚠️ This pipeline must be run entirely on the same machine. Otherwise, getting installation instructions on one device and running builds on another will not work.
> For SWE-smith, the pipeline was run on the following machine:
```bash
(swemm) john-b-yang@bitbop:~/swe-smith$ uname -m
x86_64
(swemm) john-b-yang@bitbop:~/swe-smith$ uname
Linux
```

Or, to just download the existing SWE-smith images...
```bash
python -m swesmith.build_repo.download_images
```