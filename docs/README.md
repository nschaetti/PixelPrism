# PixelPrism Documentation

This directory contains the documentation for PixelPrism, generated using MkDocs.

## Building the Documentation

To build the documentation, you need to install the required dependencies first:

```bash
# Install the package with documentation dependencies
pip install -e ".[docs]"
```

Then you can build the documentation using MkDocs:

```bash
# Build the documentation
mkdocs build

# Or serve it locally
mkdocs serve
```

## Documentation Structure

- `index.md`: The main landing page for the documentation
- `examples.md`: Examples of how to use PixelPrism
- `reference/`: Auto-generated API reference documentation
- `gen_ref_pages.py`: Script that generates the API reference documentation

## How Documentation is Generated

The documentation is generated using:

1. **mkdocs**: The main documentation generator
2. **mkdocs-material**: Theme for the documentation
3. **mkdocstrings**: Generates documentation from docstrings
4. **mkdocs-gen-files**: Generates documentation files from code
5. **mkdocs-literate-nav**: Creates navigation from Markdown files
6. **mkdocs-autorefs**: Automatically creates references between symbols

The `gen_ref_pages.py` script scans the `pixel_prism` package and generates Markdown files for each module, which are then processed by MkDocs to create the final documentation.

## Adding New Documentation

To add new documentation:

1. Add or update docstrings in the code
2. Create new Markdown files in the `docs/` directory
3. Update the `nav` section in `mkdocs.yml` if needed
4. Build the documentation to see the changes