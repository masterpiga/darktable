# color harmonizer

**Group:** color
**Pipeline position:** linear RGB, scene-referred
**Internal color space:** JzAzBz / JzCzhz (perceptual)

---

## What it does

The color harmonizer nudges hues toward a selected color palette — a set of geometrically related
hue angles called *harmony nodes* — while leaving luminance and chroma untouched. The goal is to
reduce chromatic discord in an image: colors that are "off-palette" are gently pulled toward the
nearest node; colors already on a node are left in place.

The effect is entirely in the hue dimension. Brightness, saturation, and tone relationships are
preserved.

---

## Color science

### Color space: JzCzhz

All processing happens in **JzCzhz**, the polar form of the **JzAzBz** perceptual color space
(Safdar et al., 2017, *Optics Express* 25(13)).

JzAzBz is built on the PQ (SMPTE ST 2084) electro-optical transfer function and is designed to
be perceptually uniform across the full SDR and HDR luminance range. Its polar decomposition gives:

| Channel | Symbol | Meaning |
|---------|--------|---------|
| Lightness | Jz | Perceptual brightness, ~ 0 (black) to 0.4+ (very bright) |
| Chroma | Cz | Colorfulness, ~ 0 (gray) to 0.5 (fully saturated) |
| Hue | hz | Hue angle, normalized to [0, 1] (0° – 360°) |

The pipeline for each pixel is:

```
linear RGB (D50)
  → XYZ D50
  → XYZ D65          (Bradford chromatic adaptation)
  → JzAzBz           (PQ non-linearity + matrix)
  → JzCzhz           (cartesian → polar)
  [hue modified]
  → JzAzBz           (polar → cartesian)
  → XYZ D65
  → XYZ D50          (inverse chromatic adaptation)
  → linear RGB (D50)
```

Only `hz` is ever modified. `Jz` and `Cz` pass through unchanged.

### Hue normalization

Hue angles are stored as fractions of a full rotation ([0, 1] = [0°, 360°]). All angular
arithmetic wraps correctly at the 0/1 boundary.

### Gaussian hue attraction

Each pixel is pulled toward its **nearest** harmony node. The strength of that pull is weighted
by a Gaussian whose width is controlled by the zone-width factor.

```
σ = zone_width_factor × 0.5 / N          (N = number of nodes)

w_i       = exp(−d_i² / 2σ²)             (d_i = circular distance from h to node i)
diff_i    = nodes[i] − h  (wrapped to [−0.5, 0.5])

nearest   = argmax_i(w_i)
hue_shift = w_nearest × diff_nearest
```

The Gaussian weight `w_nearest` acts as a **proximity gate**: at wide zone widths it stays near 1
across the entire hue circle so all pixels are attracted; at narrow widths it falls off quickly so
only hues close to a node are affected. The direction of pull is always toward the single nearest
node, avoiding the cancellation artefact that occurs in a weighted-average when opposing nodes
(e.g. complementary) pull in opposite directions and partially neutralize each other.

### The applied shift

```
cutoff        = t³ · 0.3                          t = protect_neutral ∈ [0, 1]
chroma_weight = Cz / (Cz + cutoff + ε)

pull    = effect_strength × chroma_weight
new_hue = h + hue_shift × pull
        = h + (w_nearest × diff_nearest) × pull
```

The cubic exponent on `t` distributes slider sensitivity evenly: small values of
`protect_neutral` affect only near-absolute grays; large values reach into muted, pastel colors.

---

## Harmony rules

All node positions are offsets from the anchor hue. Angles are in degrees.

| Rule | Nodes | Node positions | Character |
|------|-------|----------------|-----------|
| **Monochromatic** | 1 | 0° | Single hue family |
| **Analogous** | 3 | 0°, −30°, +30° | Adjacent neighbors; naturalistic, cohesive |
| **Analogous complementary** | 4 | 0°, −30°, +30°, +180° | Analogous triad plus its opposite |
| **Complementary** | 2 | 0°, +180° | Direct opposites; maximum contrast |
| **Split complementary** | 3 | 0°, +150°, +210° | Anchor plus both neighbors of its complement |
| **Dyad** | 2 | −30°, +30° | Anchor is symmetry axis, not a node |
| **Triad** | 3 | 0°, +120°, +240° | Evenly spaced; balanced, colorful |
| **Tetrad** | 4 | −30°, +30°, +150°, +210° | Two dyad pairs; anchor is symmetry axis, not a node |
| **Square** | 4 | 0°, +90°, +180°, +270° | Four equally spaced hues |

> **Note on dyad and tetrad:** the anchor hue sets the symmetry axis of the pattern, not the
> position of a node. The palette is symmetric around the anchor. This matches the vectorscope
> harmony guide overlay.

---

## Controls

### Harmony rule
Selects the geometric pattern of the target palette. See the table above.

### Anchor hue
The primary hue from which all node positions are derived. Expressed as a normalized value
displayed in degrees. An eyedropper is available to sample a color directly from the image;
colors with Cz < 0.005 are rejected as too neutral to yield a meaningful hue angle.

The color swatch strip below the slider shows the actual node colors at a fixed Jz and Cz for
quick visual feedback.

### Auto detect
Analyses the preview image's chroma-weighted hue histogram and automatically selects the harmony
rule and anchor hue that best fit the image's existing color distribution — i.e. the combination
that already covers the most chromatic energy and therefore requires the least correction.

The algorithm:
1. Builds a 360-bin hue histogram weighted by chroma (achromatic pixels are ignored).
2. Smooths it with three passes of a circular box filter to suppress noise.
3. Scores all 9 × 72 = 648 (rule, anchor) combinations at 5° resolution by computing what
   fraction of chromatic energy falls within the Gaussian attraction zones of each rule's nodes.
4. Sets the rule and anchor hue to the combination with the highest coverage score.

The result replaces the current rule and anchor. Use **effect strength** to control how strongly
the remaining off-palette hues are then pulled toward the detected palette.

### Set from vectorscope
Imports the harmony rule and anchor hue currently configured in the vectorscope panel, then
switches the histogram panel to the vectorscope view.

### Keep vectorscope in sync
When enabled (default), any change to the harmony rule or anchor hue is immediately reflected in
the vectorscope harmony overlay. Disable to make adjustments without disturbing the vectorscope
display.

### Effect strength
Global scale on the hue pull. At 0 nothing changes. At 1 the Gaussian proximity weight is the
only limit on how far each pixel moves; a pixel exactly on a node shifts by 0, a pixel at the
edge of the attraction zone shifts by the Gaussian weight multiplied by its angular distance to
that node.

### Effect width
Scales the standard deviation σ of each node's Gaussian attraction zone. Range: 0.25–4.0.

- **< 1 (narrow):** the Gaussian decays quickly with distance; only hues very close to a node
  are attracted. The rest of the hue wheel is barely touched. Useful for images already close to
  harmonic, or when precise, surgical correction of specific hues is needed.
- **= 1 (default):** the Gaussian tapers to roughly 14 % at the midpoint between adjacent nodes —
  clean zone separation with smooth transitions.
- **> 1 (wide):** the Gaussian stays high across most of the hue circle; all pixels are
  attracted noticeably regardless of how far they are from a node. Useful for strongly discordant
  images or a painterly look.

### Protect neutral
Shields low-chroma pixels from correction. The weight for each pixel is:

```
chroma_weight = Cz / (Cz + t³ · 0.3)
```

At Cz = 0 the weight is always zero regardless of the slider: fully achromatic pixels (pure
grays) are never touched. As Cz grows, the weight approaches 1. The slider sets how aggressively
low-chroma pixels are exempted: low values protect only near-absolute grays; high values extend
protection to muted and pastel tones.

Default: 0.20.

---

## Usage guide

### Basic workflow

1. **Set the anchor hue.** Use the eyedropper to pick the dominant or most important color in
   the scene — a sky, a garment, a skin tone — or drag the slider manually while watching the
   swatch strip.

2. **Choose a harmony rule.** For portraiture, *complementary* (subject vs. background) or
   *analogous* (warm tones) are natural starting points. For landscapes, *triad* or *split
   complementary* often work well. Enable *keep vectorscope in sync* to see the palette on the
   vectorscope while you choose.

3. **Raise effect strength slowly.** Start around 0.2–0.3. The effect is graduated; values
   above 0.6 are rarely needed for a convincing result.

4. **Adjust effect width if needed.** If many off-palette hues are not moving enough, widen the
   zones. If you want to correct only specific hue ranges without touching the rest, narrow them.

5. **Use protect neutral to taste.** The default 0.2 protects near-grays from unwanted tinting.
   Raise it for images with many desaturated tones (architecture, faded film looks).

### Working with the vectorscope

With *keep vectorscope in sync* enabled, the vectorscope overlays the harmony guide on the
image's color distribution. Colors inside the guide arcs are on-palette; colors outside are
being corrected. Use this to verify the pull is moving in the intended direction.

To start from a palette already set in the vectorscope panel, click *set from vectorscope*.

### Interaction with blending

The module supports parametric and drawn masking. Common uses:

- Apply harmonization only to the background by masking out the subject.
- Use a luminance mask to restrict corrections to midtones.
- Reduce opacity as a global intensity control on top of effect strength.

### Typical parameter ranges

| Goal | Effect strength | Effect width | Protect neutral |
|------|----------------|--------------|-----------------|
| Subtle finishing touch | 0.1–0.25 | 1.0–1.5 | 0.2–0.4 |
| Moderate creative grade | 0.3–0.5 | 1.5–2.5 | 0.2–0.5 |
| Aggressive stylization | 0.5–0.9 | 2.5–4.0 | 0.0–0.3 |
| Fix specific off-hues only | 0.4–0.8 | 0.25–0.7 | 0.2 |

---

## Technical notes

- The module operates entirely on hue and introduces **no luminance or chroma change** to any
  pixel. All radiometric accuracy (exposure, tone, saturation) is preserved.
- OpenCL acceleration is supported; the GPU path is numerically equivalent to the CPU path.
- Hue is undefined for fully achromatic pixels (Cz = 0). These pixels are unaffected by design:
  the chroma weight drives to zero regardless of other settings.
- The JzAzBz conversion expects absolute luminance in cd/m². Because darktable's pipeline
  carries scene-referred linear values that are not calibrated to absolute luminance, the space
  is used in a relative sense. Its perceptual uniformity and hue-angle stability are still
  superior to HSL or LCh(ab) for this purpose, but behavior at extreme HDR luminance values
  may differ from the space's theoretical specification.
