# Cross-platform Workflows

scAtlasPy can keep the complete Atlas as the persistent foundation for an
analysis, while specialized external tools are used for focused tasks such as
annotation, trajectory analysis, regulatory-network analysis, or downstream
visualization.

These guides explain how to prepare selected Atlas data for external methods,
return external results to the Atlas, and integrate third-party methods with
scAtlasPy's query and streaming interfaces.

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Use External Methods
:link: use-external-methods-on-atlas-subsets
:link-type: doc

Prepare selected cells, genes, metadata, and expression values for specialized
methods that expect an in-memory object or exported file.
:::

:::{grid-item-card} Return External Results
:link: return-external-results-to-atlas
:link-type: doc

Bring labels, scores, embeddings, gene statistics, or other outputs from an
external method back into the original Atlas.
:::

:::{grid-item-card} Integrate Third-party Methods
:link: integrate-third-party-methods
:link-type: doc

Adapt a method to access Atlas data through SQL queries, selected subsets,
single-pass streams, or randomized minibatches.
:::

::::

## Choose a Guide

| Your goal | Guide |
|---|---|
| Apply a specialized method to selected cells or genes | {doc}`use-external-methods-on-atlas-subsets` |
| Add external labels, scores, embeddings, or statistics back to the Atlas | {doc}`return-external-results-to-atlas` |
| Adapt a third-party method to access Atlas data directly | {doc}`integrate-third-party-methods` |

```{tip}
For examples of developing new algorithms with Atlas streams or minibatches,
see the {doc}`../tutorials/advanced/index`.
```

```{toctree}
:hidden:
:maxdepth: 1

Use External Methods <use-external-methods-on-atlas-subsets>
Return External Results <return-external-results-to-atlas>
Integrate Third-party Methods <integrate-third-party-methods>
```
