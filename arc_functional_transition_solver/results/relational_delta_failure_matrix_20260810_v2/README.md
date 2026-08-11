# Relational-delta v0.2 terminal failure matrix v2

This directory supersedes the retained
`../relational_delta_failure_matrix_20260810/` machine matrix after the
diagnostic script was passed through the repository formatter.

- matrix ID:
  `ce11afec96e717dd44a2c4354a44eb84fdc523b515f5bf62c3d23ee0906ddf3a`;
- matrix JSON SHA-256:
  `72f12424dbf8adc5de4c4410cec1ee158f1edde12fcb4c1a73a566e407b8feb1`;
- matrix CSV SHA-256:
  `d13146cb766553c747335c36680aa9fea6d6d6c15425c56e341cdb94d6332f88`;
- stderr SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

All 48 task records and category counts are exactly equal to v1, and the CSV is
byte-identical.  Only the content-addressed source-contract hash and resulting
matrix identity changed.  The retained v1 directory contains the identical
matrix plus demonstration-only visual review sheets.

The result remains `18 parse_failure / 14 relation_missing / 12
canvas_incompatible / 3 legal_but_inexact / 1 ast_insufficient`.  Query gold and
ARC-AGI-2 public evaluation were not read.
