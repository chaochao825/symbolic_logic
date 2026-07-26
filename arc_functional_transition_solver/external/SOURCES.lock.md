# External source lock

Third-party repositories and dataset working trees are intentionally excluded from
the public Git subtree. Recreate them from these pinned origins and commits.

| Source | Origin | Commit | Upstream license | Published license text |
|---|---|---|---|---|
| ARC-AGI-1 | `https://github.com/fchollet/ARC-AGI.git` | `399030444e0ab0cc8b4e199870fb20b863846f34` | Apache-2.0 | `ARC-AGI-1/LICENSE` |
| ARC-AGI-2 | `https://github.com/arcprize/ARC-AGI-2.git` | `f3283f727488ad98fe575ea6a5ac981e4a188e49` | Apache-2.0 | `ARC-AGI-2/LICENSE` |
| ARC-AGI benchmarking | `https://github.com/arcprize/arc-agi-benchmarking.git` | `688c9be4fb270ad7eede09fe8bba6f8187be3bed` | MIT | `arc-agi-benchmarking/LICENSE.md` |
| RE-ARC | `https://github.com/michaelhodel/re-arc.git` | `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce` | MIT | `re-arc/LICENSE` |
| Tiny Recursive Models | `https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git` | `c01103738605ba39d1430519b1ee0c62f4c707f8` | MIT | audited upstream; not vendored |
| Denoising Recursion Models | `https://github.com/wwwwwwwwz/DenoisingRecursionModels.git` | `e65d15dc41551b4bf9ae3365c7ec20842c8f06e2` | MIT | audited upstream; not vendored |
| Universal Reasoning Model | `https://github.com/zitian-gao/URM.git` | `52ee34b62fe68ef00d2784e4d4addfcd8c2a7615` | no license file found at pin | audited upstream; not vendored |
| mdlARC | `https://github.com/mvakde/mdlARC.git` | `8afc20decf8f906403193e50d23030a2413c2f7a` | MIT | audited upstream; not vendored |
| CompressARC | `https://github.com/iliao2345/CompressARC.git` | `83a22218024d46273eb32b769a906340202ffb4d` | MIT | audited upstream; not vendored |

No third-party `.git` directory, public-evaluation payload, generated RE-ARC cache,
or sanitized data ZIP is vendored by the GitHub publication builder. Existing
content-addressed manifests in `results/` retain the data provenance commitments.
The upstream license texts above are copied byte-for-byte from the pinned working
trees so that published grid evidence is accompanied by its source license terms.
The recursive-reasoning repositories are pinned research references only. No code,
weights, or data from them are copied into this project.
