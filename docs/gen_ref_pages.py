"""Generate the code reference pages and navigation."""

from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Path to the source code
src_dir = Path("pixel_prism")
# Path in the documentation where the reference will be generated
reference_dir = Path("reference")

# Iterate through all Python files in the source directory
for path in sorted(src_dir.rglob("*.py")):
    module_path = path.relative_to(Path(".")).with_suffix("")
    doc_path = path.relative_to(Path(".")).with_suffix(".md")
    full_doc_path = reference_dir / doc_path

    parts = tuple(module_path.parts)

    # Skip __init__.py files for navigation
    if parts[-1] == "__init__":
        continue

    # Create the navigation structure
    nav[parts] = doc_path.as_posix()

    # Generate the Markdown file with the proper heading
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"# {identifier}\n\n")
        fd.write(f"::: {identifier}\n")

    # Create an __init__.py file in the docs directory to make it a package
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Generate the navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())