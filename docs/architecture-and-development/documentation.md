# Documentation Maintenance

This page explains how to update and build the scAtlasPy documentation. It is
intended for documentation maintainers and contributors rather than general
users.

User-facing documentation should prioritize researchers who may not have a
computer-science background. Explain the scientific or analytical task first,
then show the code, and clearly identify the values users need to replace.

## Writing Principles

### Organize by User Task

Tutorials should follow analysis tasks and research workflows rather than the
layout of the source code.

For example, prefer:

```text
Import data → quality control → preprocessing → clustering → annotation
```

over separate pages organized only by Python modules.

### Explain Before Showing Code

Before each code block, briefly explain:

- what the step accomplishes;
- why it is needed;
- what result it creates in the Atlas.

After the code block, explain any parameters or names that users are expected
to replace.

### Keep Examples Easy to Adapt

Python examples may include short comments where they help readers understand
the workflow:

```python
# Select quality-controlled T cells
selected = obs[
    obs["filter_cells"].fillna(False)
    & obs["cell_type_manual"].eq("T cell")
]
```

Avoid comments that merely repeat the code.

### Present scAtlasPy Workflows Clearly

Describe workflows according to how users complete tasks in scAtlasPy, rather
than structuring the documentation mainly as a translation from another
platform.

Comparisons with AnnData, SeuratObject, Scanpy, Seurat, or other ecosystems can
be useful when they help readers understand a concept. In comparison tables or
workflow descriptions, introduce the scAtlasPy representation first, then show
the closest corresponding concepts in other platforms.

The goal is not to imply that one platform replaces another, but to make the
role of scAtlasPy clear while helping readers connect it with tools they may
already know.

### Separate Tutorials from API Reference

Tutorials should explain how to complete a workflow. API pages should document
public classes, functions, parameters, return values, prerequisites, and stored
outputs.

Do not copy complete API descriptions into tutorials.

### Document Persistent Outputs

Many scAtlasPy functions write results to the Atlas rather than returning a new
in-memory object. Documentation should state:

- what is returned to Python;
- which Atlas columns, fields, or tables are created;
- whether an existing result may be replaced;
- which later steps depend on that result.

### Avoid Unsupported Claims

Do not describe experimental, incomplete, or unsupported behavior as stable.
Current restrictions should be recorded in
{doc}`known-limitations`.

## Documentation Sources

Do not migrate documentation content from the obsolete root-level `docs/`
directory.

New and revised content should be written directly in the current documentation
source tree.

Before moving older text, verify that it still matches:

- the current public API;
- current parameter names and defaults;
- the current Atlas data model;
- the current supported workflows.

## Maintaining the API Reference

When adding, renaming, or removing a public function, update the corresponding
page under `api/`.

The API reference should reflect objects that are actually exported through the
package namespaces, including:

```text
scatlaspy/__init__.py
scatlaspy/io/__init__.py
scatlaspy/pp/__init__.py
scatlaspy/tl/__init__.py
scatlaspy/pl/__init__.py
```

Do not document internal helpers as public functions unless they are
intentionally exported and supported.

For each public interface, document:

- its purpose;
- parameters and accepted values;
- required Atlas state;
- Python return value;
- results written to the Atlas;
- replacement or overwrite behavior;
- a minimal example.

After changing a docstring or API page, rebuild the documentation and check the
generated reference page.

## Links and Navigation

Use MyST document links for internal pages:

```markdown
{doc}`../tutorials/index`
```

Add new pages to the appropriate `toctree`; otherwise, they may not appear in
the website navigation.

Check that:

- page paths are correct;
- headings are descriptive;
- links do not point to obsolete pages;
- the same content is not duplicated across tutorials and API pages.

## Build the Documentation

Only contributors modifying the website need to install the documentation
dependencies.

From the repository root:

```bash
python -m pip install -e ".[docs]"
```

Build the HTML documentation:

```bash
cd docs
make html
```

The generated website is written to:

```text
docs/_build/html/
```

Treat warnings as potential documentation problems, especially warnings about:

- missing references;
- pages not included in a `toctree`;
- failed API imports;
- duplicated labels;
- malformed MyST or reStructuredText directives.

## Preview Locally

Start the local documentation server with:

```bash
make serve
```

The server listens on port `8010` by default.

Open the site locally at:

```text
http://127.0.0.1:8010/
```

When the server is bound to `0.0.0.0`, other computers on the same network can
access it through the host machine's IP address, for example:

```text
http://192.168.1.23:8010/
```

`0.0.0.0` is the listening address and should not normally be used as the URL in
a browser.

Use another port when the default port is occupied:

```bash
make serve PORT=8020
```

## Review Before Submitting

Before submitting documentation changes:

1. Build the website without unexpected warnings.
2. Open every modified page in the browser.
3. Check code blocks, tables, notes, and internal links.
4. Confirm that examples use the current public API.
5. Verify that tutorials remain understandable without reading the source code.
6. Confirm that API pages accurately describe stored outputs and prerequisites.
7. Check that new pages appear in the intended navigation section.
