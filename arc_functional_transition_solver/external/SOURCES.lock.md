# External source lock

Third-party repositories and dataset working trees are intentionally excluded from
the public Git subtree. Recreate them from these pinned origins and commits.

| Source | Origin | Commit | Upstream license | Published license text |
|---|---|---|---|---|
| ARC-AGI-1 | `https://github.com/fchollet/ARC-AGI.git` | `399030444e0ab0cc8b4e199870fb20b863846f34` | Apache-2.0 | `ARC-AGI-1/LICENSE` |
| ARC-AGI-2 | `https://github.com/arcprize/ARC-AGI-2.git` | `f3283f727488ad98fe575ea6a5ac981e4a188e49` | Apache-2.0 | `ARC-AGI-2/LICENSE` |
| ARC-AGI benchmarking | `https://github.com/arcprize/arc-agi-benchmarking.git` | `688c9be4fb270ad7eede09fe8bba6f8187be3bed` | MIT | `arc-agi-benchmarking/LICENSE.md` |
| RE-ARC | `https://github.com/michaelhodel/re-arc.git` | `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce` | MIT | `re-arc/LICENSE` |

No third-party `.git` directory, public-evaluation payload, generated RE-ARC cache,
or sanitized data ZIP is vendored by the GitHub publication builder. Existing
content-addressed manifests in `results/` retain the data provenance commitments.
The upstream license texts above are copied byte-for-byte from the pinned working
trees so that published grid evidence is accompanied by its source license terms.
