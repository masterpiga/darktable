# JzCzhz CPU/OpenCL precision harnesses

Standalone programs behind the numbers in
`upstreamed_rendering_fixes_from_flexi_migration.md`, finding 2. Each contains
**verbatim copies** of darktable's host implementation
(`src/common/colorspaces_inline_conversions.h`, `src/common/iop_profile.h`) and
of its OpenCL counterpart (`data/kernels/colorspace.h`), fed identical inputs.
They have no darktable dependency and are not part of the build.

```
clang -O2 -Wno-deprecated-declarations <file>.c -framework OpenCL -o <file>
```

(on non-Apple platforms, link the OpenCL SDK instead of the framework)

| file | question it answers |
|---|---|
| `jzdiff.c` | which JzCzhz channel disagrees, and at what chroma |
| `anchor.c` | what a given Cz means in visible terms (neutral patch, one 8-bit step) |
| `stage.c` | at which stage of XYZ -> JzAzBz the disagreement enters |
| `powtest.c` | whether a more accurate kernel-side `pow` would help (it would not) |
| `stable.c` | how much the proposed cancellation-free formulation improves it, and how much it changes the existing render |

If the shipped conversions change, re-copy the code into these files rather
than editing them in place -- the point is that both sides are the real thing.
