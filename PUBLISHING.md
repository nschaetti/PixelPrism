# Publishing PixelPrism to PyPI

This document provides instructions for publishing the PixelPrism package to the Python Package Index (PyPI), making it available for installation via pip.

## Prerequisites

1. Create accounts on PyPI and Test PyPI:
   - PyPI: https://pypi.org/account/register/
   - Test PyPI: https://test.pypi.org/account/register/

2. Install required tools:
   ```bash
   pip install build twine
   ```

## Building the Package

1. Make sure your working directory is clean (no uncommitted changes).

2. Update the version number in `setup.py` if necessary.

3. Build the package:
   ```bash
   python -m build
   ```

   This will create both source distribution (.tar.gz) and wheel (.whl) files in the `dist/` directory.

## Testing the Package

Before publishing to the main PyPI repository, it's recommended to test the package on Test PyPI:

1. Upload to Test PyPI:
   ```bash
   python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

2. Install from Test PyPI to verify it works:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pixelprism
   ```

3. Test that the installed package works correctly.

## Publishing to PyPI

Once you've verified that the package works correctly, you can publish it to the main PyPI repository:

1. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

2. Verify the installation:
   ```bash
   pip install pixelprism
   ```

## Updating the Package

When you need to update the package:

1. Make your changes to the codebase.

2. Update the version number in `setup.py`.

3. Rebuild the package:
   ```bash
   python -m build
   ```

4. Upload the new version to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

## Additional Resources

- [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/)
- [PyPI Publishing Documentation](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
- [Twine Documentation](https://twine.readthedocs.io/en/latest/)