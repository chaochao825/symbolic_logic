# Innovation Candidate: Boolean/Circuit-MDL Rate Reduction

## Surviving framing

Continuous coding-rate reduction and Boolean circuit complexity are not the same quantity. The defensible bridge is a multi-part prefix code:

```text
continuous geometry
  -> universal Boolean representation code
  -> route-coded circuit language
  -> combinatorial task residual
  -> optional technology-aware constraint
```

The candidate contribution is not a new language-free Boolean entropy. It is an executable protocol that makes every claimed bit decodable and exposes language, basis, side-information, and synthesis assumptions.

## Falsifiable claims

1. A routed prefix code satisfies Kraft and never loses to its raw escape path.
2. Independent-coordinate KT is basis dependent, while a joint codeword mixture is invariant to codeword relabeling.
3. Exact four-input formula enumeration separates parity from the balanced-function population although output entropy is identical.
4. Multi-language MDL selects ANF for parity, threshold code for majority, and raw labels for typical random LUTs.

## Killed framings

- MCR² bits directly equal Boolean storage bits or gate count.
- One heuristic gate count is a complexity oracle.
- A fixed AND/OR/XOR/NAND library is optimal for every Boolean function.
- Hardware cost can be added to Shannon bits without an explicit exchange coefficient.

## Reviewer-facing limits

- Exactness is formula-tree exactness, not minimum DAG.
- Small truth tables validate ordering and certificates but may not have positive net compression after headers.
- The benchmark is synthetic and finite; natural perceptual representations remain future work.
- A data-selected new grammar must itself be coded or selected on an independent split.
