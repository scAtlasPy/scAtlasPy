# Architecture and Development

This section is intended for readers who want to understand how scAtlasPy
works internally and for contributors who want to fix issues, improve existing
modules, or add new capabilities.

Most users do not need these pages to analyze data. For normal usage, start with
{doc}`../installation` and the {doc}`../tutorials/index`.

## Who Should Read This Section?

This section is primarily intended for:

- users who want to understand the storage model, streaming computation, and
  performance characteristics of scAtlasPy;
- contributors who want to modify the codebase, documentation, or supported
  workflows;
- developers evaluating scAtlasPy as a foundation for atlas-scale methods.

## Recommended Reading

| Goal | Page |
|---|---|
| Understand the tables and data structures stored in an Atlas | {doc}`data-model` |
| Understand single-pass and multi-pass minibatches and the shuffle buffer | {doc}`minibatch-architecture` |
| Tune data import, batch retrieval, and analysis performance | {doc}`performance` |
| Modify or extend the documentation website | {doc}`documentation` |
| Review current limitations and unsupported workflows | {doc}`known-limitations` |

A useful reading order for understanding the system is:

1. {doc}`data-model`
2. {doc}`minibatch-architecture`
3. {doc}`performance`
4. {doc}`known-limitations`

Contributors working only on documentation can begin directly with
{doc}`documentation`.

```{toctree}
:hidden:
:maxdepth: 1

data-model
minibatch-architecture
performance
documentation
known-limitations
```
