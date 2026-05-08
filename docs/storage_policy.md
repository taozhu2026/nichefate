# Storage Policy

`/home/zhutao/projects/nichefate` is the code root. Keep source code,
lightweight configs, documentation, and tests there.

Large raw data and long-lived heavy intermediate data belong under the organized
project data root:

```text
/data/zhutao/nichefate
```

The intended layout is:

```text
/data/zhutao/nichefate/
  raw/
  m0/
    input/
    intermediate/
    by_slice/
    reports/
  m1/
    archived_or_heavy/
  m2/
    archived_or_heavy/
  external/
  manifests/
  logs/
```

`/home/zhutao/scratch/nichefate` is for active working outputs, lightweight
reports, active small temporary files, and short-lived intermediates that are
being computed. It is not a safe default for new raw downloads or bulky
long-lived intermediates.

If an existing script expects an old `/home/zhutao/scratch/nichefate` path, keep
compatibility with a symlink or update the path through config in a separate
reviewed change. Do not rewrite algorithm logic just to move storage.

Future downloads should default to `/data/zhutao/nichefate`. Do not store new
raw data under `/home` unless it is explicitly temporary and small enough for
the available space.

Before large active computations, estimate disk usage and choose `/data` when
`/home` is insufficient. Current M4A, M4B, M4C, and M4D production outputs are
not part of the initial storage migration stage.
