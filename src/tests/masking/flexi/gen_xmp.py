#!/usr/bin/env python3
"""Generate classic-format (pre-flexi) test XMPs for the darktable
classic->flexi mask migration, covering drawn/parametric/combined
configurations with varying combine operators and polarity inversions.
"""
import struct
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "xmps")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# dt_develop_blend_params_t (blend version 14, 420 bytes, confirmed against a
# real darktable-generated default blob: identical layout to current v15/v8).
# ---------------------------------------------------------------------------
DEVELOP_MASK_ENABLED = 1
DEVELOP_MASK_MASK = 1 << 1
DEVELOP_MASK_CONDITIONAL = 1 << 2
DEVELOP_MASK_RASTER = 1 << 3
DEVELOP_MASK_MASK_CONDITIONAL = DEVELOP_MASK_MASK | DEVELOP_MASK_CONDITIONAL

DEVELOP_COMBINE_NORM = 0x00
DEVELOP_COMBINE_INV = 0x01
DEVELOP_COMBINE_INCL = 0x02
DEVELOP_COMBINE_MASKS_POS = 0x04

DEVELOP_BLEND_CS_RGB_SCENE = 4
DEVELOP_BLEND_NORMAL2 = 0x18

# RGB-scene blendif channel indices (see dt_develop_blendif_channels_t)
CH_GRAY_in = 0
CH_RED_in = 1
CH_GREEN_in = 2
CH_BLUE_in = 3
DEVELOP_BLENDIF_active = 31
DEVELOP_BLENDIF_SIZE = 16

DT_MASKS_STATE_USE = 1 << 0
DT_MASKS_STATE_SHOW = 1 << 1
DT_MASKS_STATE_INVERSE = 1 << 2
DT_MASKS_STATE_UNION = 1 << 3
DT_MASKS_STATE_INTERSECTION = 1 << 4
DT_MASKS_STATE_DIFFERENCE = 1 << 5
DT_MASKS_STATE_EXCLUSION = 1 << 6
# pre-v10 encoding of a first-class group boundary (see
# dt_masks_point_group_t.group_start in masks.h); only used by
# build_group_start_scenario() below to exercise the v9->v10 migration that
# carries this bit forward into the real field.
DT_MASKS_STATE_GROUP_BREAK = 1 << 11

DT_MASKS_CIRCLE = 1
DT_MASKS_PATH = 1 << 1
DT_MASKS_GROUP = 1 << 2
DEVELOP_MASKS_VERSION = 8
DT_MASKS_POINT_STATE_USER = 2

BLEND_FMT = "<3i2f2ifIfIffffI2I" + "64f" + "16f" + "20siii"
# mask_mode(I) blend_cst(i) blend_mode(I) blend_parameter(f) opacity(f)
# mask_combine(I) mask_id(i) blendif(I) feathering_radius(f)
# feathering_guide(I) blur_radius(f) contrast(f) brightness(f) details(f)
# feather_version(I) reserved[2](II) blendif_parameters[64](f)
# blendif_boost_factors[16](f) raster_mask_source[20](s)
# raster_mask_instance(i) raster_mask_id(i) raster_mask_invert(i)
BLEND_FMT = "<IiIffIiIfIffffI2I64f16f20siii"


def pack_blend_params(mask_mode=0, blend_cst=0,
                       blend_mode=DEVELOP_BLEND_NORMAL2, blend_parameter=0.0,
                       opacity=100.0, mask_combine=DEVELOP_COMBINE_NORM,
                       mask_id=0, blendif=0, blendif_parameters=None,
                       blendif_boost_factors=None):
    if blendif_parameters is None:
        blendif_parameters = [0.0, 0.0, 1.0, 1.0] * DEVELOP_BLENDIF_SIZE
    if blendif_boost_factors is None:
        blendif_boost_factors = [0.0] * DEVELOP_BLENDIF_SIZE
    assert len(blendif_parameters) == 4 * DEVELOP_BLENDIF_SIZE
    assert len(blendif_boost_factors) == DEVELOP_BLENDIF_SIZE
    data = struct.pack(
        BLEND_FMT,
        mask_mode, blend_cst, blend_mode, blend_parameter, opacity,
        mask_combine, mask_id, blendif,
        0.0,  # feathering_radius
        1,    # feathering_guide (DEVELOP_MASK_GUIDE_IN_BEFORE_BLUR)
        0.0, 0.0, 0.0, 0.0,  # blur_radius, contrast, brightness, details
        0,    # feather_version
        0, 0,  # reserved[2]
        *blendif_parameters,
        *blendif_boost_factors,
        b"",   # raster_mask_source
        -1,    # raster_mask_instance
        0,     # raster_mask_id
        0,     # raster_mask_invert
    )
    assert len(data) == 420, len(data)
    return data


def channel_curve(channels, taper_in=(0.0, 0.3), taper_off=(0.5, 0.8),
                   invert_channels=()):
    """Set a 0-30% taper-in / 50-80% taper-off curve on the given channel
    indices; all other channels stay at the neutral/disabled (0,0,1,1)."""
    params = [0.0, 0.0, 1.0, 1.0] * DEVELOP_BLENDIF_SIZE
    blendif = 1 << DEVELOP_BLENDIF_active
    for ch in channels:
        params[ch * 4:ch * 4 + 4] = [taper_in[0], taper_in[1], taper_off[0], taper_off[1]]
        blendif |= 1 << ch
        if ch in invert_channels:
            blendif |= 1 << (ch + 16)
    return blendif, params


# ---------------------------------------------------------------------------
# mask forms
# ---------------------------------------------------------------------------
CIRCLE_FMT = "<4f"          # center[2], radius, border
PATH_PT_FMT = "<8fI"        # corner[2], ctrl1[2], ctrl2[2], border[2], state
GROUP_MEMBER_FMT = "<iiif" + "i6f" + "128s"
# formid(i) parentid(i) state(i) opacity(f) refinement{enabled(i) details(f)
# feathering_radius(f) feathering_guide(I->i) blur_radius(f) contrast(f)
# brightness(f)} name[128]


def pack_circle(cx, cy, radius, border):
    return struct.pack(CIRCLE_FMT, cx, cy, radius, border)


def pack_path(corners, border=(0.02, 0.02)):
    out = b""
    for (x, y) in corners:
        out += struct.pack(PATH_PT_FMT, x, y, x, y, x, y,
                            border[0], border[1], DT_MASKS_POINT_STATE_USER)
    return out


def pack_group_member(formid, parentid, state, opacity=1.0,
                       refine_enabled=0, details=0.0, feathering_radius=0.0,
                       blur_radius=0.0, contrast=0.0, brightness=0.0):
    # feathering_guide is left 0 (its 0 bit pattern is identical whether the
    # reader treats this refinement slot as int or float -- not exercised
    # here, see the FMT comment above)
    data = struct.pack(GROUP_MEMBER_FMT, formid, parentid, state, opacity,
                        refine_enabled, details, feathering_radius, 0,
                        blur_radius, contrast, brightness, b"")
    assert len(data) == 172, len(data)
    return data


GROUP_MEMBER_V9_FMT = GROUP_MEMBER_FMT + "f"
# adds group_opacity(f) (masks v9) after pack_group_member's v8 layout


def pack_group_member_v9(formid, parentid, state, opacity=1.0, group_opacity=1.0,
                          **kwargs):
    data = pack_group_member(formid, parentid, state, opacity=opacity, **kwargs)
    data += struct.pack("<f", group_opacity)
    assert len(data) == 176, len(data)
    return data


CIRCLE_CX, CIRCLE_CY, CIRCLE_R, CIRCLE_BORDER = 0.45, 0.45, 0.18, 0.21
SQUARE_CORNERS = [(0.40, 0.30), (0.70, 0.30), (0.70, 0.60), (0.40, 0.60)]
# second shape pair for build_group_start_scenario(): offset to the opposite
# corner so the two groups' masks aren't near-identical
CIRCLE2_CX, CIRCLE2_CY, CIRCLE2_R, CIRCLE2_BORDER = 0.75, 0.75, 0.15, 0.18
SQUARE2_CORNERS = [(0.05, 0.65), (0.30, 0.65), (0.30, 0.90), (0.05, 0.90)]


class MaskIds:
    def __init__(self, base):
        self.circle = base + 1
        self.path = base + 2
        self.group = base + 3


def masks_history_rows(mask_num, ids, circle_state, square_state,
                       circle_opacity=1.0, circle_refine=None):
    """Returns list of (mask_id, mask_type, mask_name, mask_points_hex, mask_nb).
    circle_refine: None, or a dict of pack_group_member's refine_* kwargs --
    exercises the classic mask manager's per-shape opacity/refinement
    (dt_masks_point_group_t.opacity/.refinement), which migration must carry
    over unchanged since it reuses the drawn group's own points verbatim."""
    rows = []
    rows.append((ids.circle, DT_MASKS_CIRCLE, "circle #1",
                 pack_circle(CIRCLE_CX, CIRCLE_CY, CIRCLE_R, CIRCLE_BORDER).hex(), 1))
    rows.append((ids.path, DT_MASKS_PATH, "square #1",
                 pack_path(SQUARE_CORNERS).hex(), len(SQUARE_CORNERS)))
    members = (pack_group_member(ids.circle, ids.group, circle_state,
                                 opacity=circle_opacity,
                                 **(circle_refine or {})) +
               pack_group_member(ids.path, ids.group, square_state))
    rows.append((ids.group, DT_MASKS_GROUP, "grp exposure", members.hex(), 2))
    return [(mask_num,) + r for r in rows]


# ---------------------------------------------------------------------------
# XMP assembly
# ---------------------------------------------------------------------------
PIPELINE = [
    # (num, operation, modversion, op_params_hex, multi_name)
    (0, "colorin", 7,
     "090000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
     ""),
    (1, "colorout", 5,
     "01000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
     ""),
    (2, "gamma", 1, "0000000000000000", ""),
    (3, "flip", 2, "ffffffff", "_builtin_auto"),
]

# dt_iop_exposure_params_t (modversion 6): mode(i) black(f) exposure(f)
# deflicker_percentile(f) deflicker_target_level(f) compensate_exposure_bias(i)
# +5EV so masked/unmasked regions are unmistakably different in the
# rendered output (a faint bump makes it hard to tell a real mask shape
# from a scenario that is accidentally a no-op).
EXPOSURE_PARAMS_HEX = struct.pack("<iffffi", 0, 0.0, 5.0, 50.0, -4.0, 0).hex()

DUMMY_HASH = "0" * 32


def build_xmp(name, blend_params_bytes, masks_rows, outdir=None, exposure_enabled=True):
    exposure_num = len(PIPELINE)
    history_end = exposure_num + 1

    hist_items = []
    for (num, op, modv, params_hex, multi_name) in PIPELINE:
        default_blend = pack_blend_params().hex()
        hist_items.append(f'''     <rdf:li
      darktable:num="{num}"
      darktable:operation="{op}"
      darktable:enabled="1"
      darktable:modversion="{modv}"
      darktable:params="{params_hex}"
      darktable:multi_name="{multi_name}"
      darktable:multi_priority="0"
      darktable:blendop_version="14"
      darktable:blendop_params="{default_blend}"/>''')

    hist_items.append(f'''     <rdf:li
      darktable:num="{exposure_num}"
      darktable:operation="exposure"
      darktable:enabled="{1 if exposure_enabled else 0}"
      darktable:modversion="6"
      darktable:params="{EXPOSURE_PARAMS_HEX}"
      darktable:multi_name=""
      darktable:multi_priority="0"
      darktable:blendop_version="14"
      darktable:blendop_params="{blend_params_bytes.hex()}"/>''')

    masks_items = []
    for (mask_num, mask_id, mask_type, mask_name, points_hex, mask_nb) in masks_rows:
        masks_items.append(f'''     <rdf:li
      darktable:mask_num="{mask_num}"
      darktable:mask_id="{mask_id}"
      darktable:mask_type="{mask_type}"
      darktable:mask_name="{mask_name}"
      darktable:mask_version="{DEVELOP_MASKS_VERSION}"
      darktable:mask_points="{points_hex}"
      darktable:mask_nb="{mask_nb}"
      darktable:mask_src="0000000000000000"/>''')

    xmp = f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:exif="http://ns.adobe.com/exif/1.0/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
    xmlns:darktable="http://darktable.sf.net/"
   xmp:Rating="1"
   xmpMM:DerivedFrom="Sweep_sRGB_Linear_Half_Zip_01.tif"
   darktable:xmp_version="5"
   darktable:raw_params="0"
   darktable:auto_presets_applied="1"
   darktable:history_end="{history_end}"
   darktable:iop_order_version="2"
   darktable:history_auto_hash="{DUMMY_HASH}"
   darktable:history_current_hash="{DUMMY_HASH}">
   <darktable:masks_history>
    <rdf:Seq>
{chr(10).join(masks_items)}
    </rdf:Seq>
   </darktable:masks_history>
   <darktable:history>
    <rdf:Seq>
{chr(10).join(hist_items)}
    </rdf:Seq>
   </darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''
    path = os.path.join(outdir or OUTDIR, f"{name}.xmp")
    with open(path, "w") as f:
        f.write(xmp)
    return path


# ---------------------------------------------------------------------------
# test matrix
# ---------------------------------------------------------------------------
def op_state(op_bit, invert=False):
    s = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE | op_bit
    if invert:
        s |= DT_MASKS_STATE_INVERSE
    return s


SCENARIOS = []


def scenario(name, mask_mode, mask_combine=DEVELOP_COMBINE_NORM, mask_id=0,
             blendif=0, blendif_parameters=None, draw=None,
             circle_opacity=1.0, circle_refine=None):
    """draw: None, or (circle_op_bit_or_None, circle_invert, square_op_bit, square_invert)
    circle_opacity/circle_refine: see masks_history_rows"""
    SCENARIOS.append(dict(name=name, mask_mode=mask_mode, mask_combine=mask_combine,
                           mask_id=mask_id, blendif=blendif,
                           blendif_parameters=blendif_parameters, draw=draw,
                           circle_opacity=circle_opacity, circle_refine=circle_refine))


# A: drawn only, varying combine operator + invert
scenario("A1_union", DEVELOP_MASK_MASK,
          draw=(None, False, DT_MASKS_STATE_UNION, False))
scenario("A2_intersection", DEVELOP_MASK_MASK,
          draw=(None, False, DT_MASKS_STATE_INTERSECTION, False))
scenario("A3_difference", DEVELOP_MASK_MASK,
          draw=(None, False, DT_MASKS_STATE_DIFFERENCE, False))
scenario("A4_exclusion", DEVELOP_MASK_MASK,
          draw=(None, False, DT_MASKS_STATE_EXCLUSION, False))
scenario("A5_union_circle_inverted", DEVELOP_MASK_MASK,
          draw=(None, True, DT_MASKS_STATE_UNION, False))
scenario("A6_intersection_group_invert", DEVELOP_MASK_MASK,
          mask_combine=DEVELOP_COMBINE_INV,
          draw=(None, False, DT_MASKS_STATE_INTERSECTION, False))

# B: parametric only, varying channel count + polarity
_b1_blendif, _b1_params = channel_curve([CH_RED_in])
scenario("B1_1channel", DEVELOP_MASK_CONDITIONAL,
          blendif=_b1_blendif, blendif_parameters=_b1_params)

_b2_blendif, _b2_params = channel_curve([CH_RED_in, CH_GREEN_in])
scenario("B2_2channel", DEVELOP_MASK_CONDITIONAL,
          blendif=_b2_blendif, blendif_parameters=_b2_params)

_b3_blendif, _b3_params = channel_curve([CH_RED_in, CH_GREEN_in, CH_BLUE_in])
scenario("B3_3channel", DEVELOP_MASK_CONDITIONAL,
          blendif=_b3_blendif, blendif_parameters=_b3_params)

_b4_blendif, _b4_params = channel_curve([CH_RED_in], invert_channels=[CH_RED_in])
scenario("B4_1channel_inverted_polarity", DEVELOP_MASK_CONDITIONAL,
          blendif=_b4_blendif, blendif_parameters=_b4_params)

_b5_blendif, _b5_params = channel_curve([CH_GREEN_in, CH_BLUE_in])
scenario("B5_2channel_group_invert", DEVELOP_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INV,
          blendif=_b5_blendif, blendif_parameters=_b5_params)

# C: drawn AND parametric combined
_c1_blendif, _c1_params = channel_curve([CH_RED_in, CH_GREEN_in])
scenario("C1_union_2channel", DEVELOP_MASK_MASK_CONDITIONAL,
          blendif=_c1_blendif, blendif_parameters=_c1_params,
          draw=(None, False, DT_MASKS_STATE_UNION, False))

_c2_blendif, _c2_params = channel_curve([CH_BLUE_in], invert_channels=[CH_BLUE_in])
scenario("C2_intersection_1channel_inverted", DEVELOP_MASK_MASK_CONDITIONAL,
          blendif=_c2_blendif, blendif_parameters=_c2_params,
          draw=(None, False, DT_MASKS_STATE_INTERSECTION, False))

_c3_blendif, _c3_params = channel_curve([CH_RED_in, CH_GREEN_in, CH_BLUE_in])
scenario("C3_difference_drawn_inverted_3channel", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_MASKS_POS,
          blendif=_c3_blendif, blendif_parameters=_c3_params,
          draw=(None, False, DT_MASKS_STATE_DIFFERENCE, False))

_c4_blendif, _c4_params = channel_curve([CH_GRAY_in])
scenario("C4_union_1channel_group_invert", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INV,
          blendif=_c4_blendif, blendif_parameters=_c4_params,
          draw=(None, False, DT_MASKS_STATE_UNION, False))

# D: previously fail-closed, now-migratable "no resolvable drawn content"
# combinations -- these translate to a real classic formula (not always the
# degenerate "always zero" case D1 alone would suggest), see the derivation
# in the E-series below for the cases D-alone's naive treatment gets wrong.
_d1_blendif, _d1_params = channel_curve([CH_RED_in])
scenario("D1_maskspos_no_drawn_content", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_MASKS_POS, mask_id=0,
          blendif=_d1_blendif, blendif_parameters=_d1_params, draw=None)

_d2_blendif, _d2_params = channel_curve([CH_GREEN_in])
scenario("D2_parametric_incl", DEVELOP_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL,
          blendif=_d2_blendif, blendif_parameters=_d2_params)

# E: INCL x INV x MASKS_POS x content-resolvability cross-matrix.
#
# NOTE on what to expect: INCL XORs *every* channel's own polarity bit in
# the mask's colorspace before classic evaluates it (see
# migrate_legacy.c's _classify_conditional()). Since every scenario below
# activates only 1-2 of the colorspace's channels, that XOR always flags at
# least one untouched channel as "canceling", which makes classic
# wholesale-replace the whole mask with a flat constant (opaque or zero,
# picked by `incl != inv`) -- never a real shaped/curve mask. So *every*
# scenario in this block (E1-E6, plus D1/D2 above) is intentionally a hard
# constant, verified two ways: against a pristine (pre-migration) binary,
# and by the module_off/mask_disabled effect-check in verify_effect.sh,
# which confirms each one is bit-identical to one of those two baselines
# (not merely "faint"). Only a channel config that activates *every*
# channel of the colorspace simultaneously would escape this and produce a
# real (screen-like) mask -- that combination has no flexi equivalent and
# is intentionally left fail-closed (stays classic), so it isn't in this
# matrix at all.

# pure parametric, INCL set alone (no INV): constant, opaque (incl!=inv).
_e1_blendif, _e1_params = channel_curve([CH_RED_in])
scenario("E1_pure_incl_only", DEVELOP_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL,
          blendif=_e1_blendif, blendif_parameters=_e1_params)

# pure parametric, INCL AND INV both set: constant, zero (incl==inv).
_e2_blendif, _e2_params = channel_curve([CH_RED_in])
scenario("E2_pure_incl_and_inv", DEVELOP_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL | DEVELOP_COMBINE_INV,
          blendif=_e2_blendif, blendif_parameters=_e2_params)

# drawn (resolvable) + parametric, INCL set alone (no MASKS_POS, no INV):
# constant, opaque -- the canceling-channel constant-replace discards the
# drawn geometry too, not just the parametric curve.
_e3_blendif, _e3_params = channel_curve([CH_GREEN_in])
scenario("E3_content_incl_only", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL,
          blendif=_e3_blendif, blendif_parameters=_e3_params,
          draw=(None, False, DT_MASKS_STATE_UNION, False))

# drawn (resolvable) + parametric, INCL AND MASKS_POS both set: constant,
# opaque -- MASKS_POS plays no role in the canceling-channel classification
# (only INCL does), so this is the same outcome as E3.
_e4_blendif, _e4_params = channel_curve([CH_RED_in, CH_GREEN_in])
scenario("E4_content_incl_maskspos", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL | DEVELOP_COMBINE_MASKS_POS,
          blendif=_e4_blendif, blendif_parameters=_e4_params,
          draw=(None, False, DT_MASKS_STATE_UNION, False))

# drawn+parametric, NO resolvable drawn content, MASKS_POS AND INCL both
# set -- exactly the state reachable via one click of "invert all
# channel's polarities" followed by deleting the drawn mask. Delegates to
# the pure-parametric path (masks_pos == incl), which again lands on
# constant/opaque per the same canceling-channel rule as E1.
_e5_blendif, _e5_params = channel_curve([CH_BLUE_in])
scenario("E5_nocontent_maskspos_and_incl", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_MASKS_POS | DEVELOP_COMBINE_INCL, mask_id=0,
          blendif=_e5_blendif, blendif_parameters=_e5_params, draw=None)

# drawn+parametric, NO resolvable drawn content, INCL set but MASKS_POS NOT
# set: constant, opaque (opposite of D1's constant/zero), regardless of
# the channel configuration.
_e6_blendif, _e6_params = channel_curve([CH_RED_in])
scenario("E6_nocontent_incl_only_opaque", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL, mask_id=0,
          blendif=_e6_blendif, blendif_parameters=_e6_params, draw=None)

# F: drawn (resolvable) + parametric, INCL set, with EVERY channel of the
# RGB-scene colorspace active at once (DEVELOP_BLENDIF_RGB_MASK's full
# channel set: GRAY/RED/GREEN/BLUE in+out, plus Jz/Cz/hz in+out) -- the one
# combination that reaches DT_COND_REAL despite INCL (no channel is left
# untouched for INCL's polarity-XOR to flag as canceling), previously
# fail-closed. Classic's formula there is 1-(1-d)*temp (INV=0, F1) or
# (1-d)*temp (INV=1, F2) -- see the DT_COND_REAL/INCL derivation in
# migrate_legacy.c's _migrate_drawn_and_parametric().
# must match DEVELOP_BLENDIF_RGB_MASK (src/develop/blend.h) exactly: bits
# 0-10, 12-14 (11 and 15 unused/reserved).
_RGB_ALL_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
assert sum(1 << c for c in _RGB_ALL_CHANNELS) == 0x77FF

_f_blendif, _f_params = channel_curve(_RGB_ALL_CHANNELS)
scenario("F1_content_incl_allchannels_noinv", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL,
          blendif=_f_blendif, blendif_parameters=_f_params,
          draw=(None, False, DT_MASKS_STATE_UNION, False))

scenario("F2_content_incl_allchannels_inv", DEVELOP_MASK_MASK_CONDITIONAL,
          mask_combine=DEVELOP_COMBINE_INCL | DEVELOP_COMBINE_INV,
          blendif=_f_blendif, blendif_parameters=_f_params,
          draw=(None, False, DT_MASKS_STATE_UNION, False))

# G: bare DEVELOP_MASK_ENABLED (classic "uniform", no MASK/CONDITIONAL/
# RASTER bit at all) -- migration deliberately never touched this before
# Phase 0.5 (it already rendered identically to an empty flexi group), but
# now normalizes it to ENABLED|FLEXI explicitly so no raw classic value is
# ever left in blend_params. Passing mask_mode=0 to scenario() below yields
# exactly DEVELOP_MASK_ENABLED once ORed with it in main().
scenario("G1_bare_uniform", 0)

# H: per-shape opacity + refinement (dt_masks_point_group_t.opacity/
# .refinement) on the drawn circle, as the classic mask manager's per-shape
# panel used to set (see src/libs/masks.c, deleted; still readable via git
# history). Migration reuses the drawn group's own points verbatim (see
# _migrate_drawn_and_parametric's "MASKS_POS moves onto the wrapper entry"
# comment), so these must survive completely unchanged -- both drawn-only
# (no rebuild at all involved) and drawn+parametric (the group gets a new
# *wrapper* point referencing it, but the drawn group's own points are still
# untouched).
_h_refine = dict(refine_enabled=1, details=0.3, feathering_radius=12.0,
                 blur_radius=8.0, contrast=0.25, brightness=-0.15)
scenario("H1_drawn_opacity_refinement", DEVELOP_MASK_MASK,
          draw=(None, False, DT_MASKS_STATE_UNION, False),
          circle_opacity=0.6, circle_refine=_h_refine)

_h2_blendif, _h2_params = channel_curve([CH_RED_in, CH_GREEN_in])
scenario("H2_combined_opacity_refinement", DEVELOP_MASK_MASK_CONDITIONAL,
          blendif=_h2_blendif, blendif_parameters=_h2_params,
          draw=(None, False, DT_MASKS_STATE_UNION, False),
          circle_opacity=0.6, circle_refine=_h_refine)


# I: flexi-native regression (NOT a classic-migration scenario like A-H
# above) for the masks v9->v10 migration that replaces the temporary
# DT_MASKS_STATE_GROUP_BREAK bit with the real dt_masks_point_group_t.
# group_start field (see masks.h / dt_masks_legacy_params_v9_to_v10 in
# masks/masks.c). Writes a single DT_MASKS_GROUP form, masks version 9
# (pre-group_start), with FOUR members forming two separate two-shape runs
# that share the same between-group operator (INTERSECTION): circleA+squareA
# as the bottom run, circleB+squareB as the run above it, with the old
# GROUP_BREAK bit set on circleB (the upper run's own bottom/head member) so
# it stays a distinct run despite matching circleA/squareA's operator.
#
# INTERSECTION is deliberately used (not UNION) because it's the one
# operator where "two separate same-op runs" and "one merged run" produce
# different pixels: two intersect-groups compute
# intersect(unionA, unionB), scoped as INDEPENDENT runs, while one merged
# 4-member group computes intersect(unionA union unionB) -- mathematically
# different in general. A wrong/no-op migration (group_start never set, or
# read back as 0) collapses this into one merged run and silently changes
# the rendered mask; this scenario's whole job is to make that collapse
# visible as a pixel diff.
def build_group_start_scenario():
    ids = MaskIds(999000)
    circleA_id, squareA_id, circleB_id, squareB_id, group_id = (
        ids.circle, ids.path, ids.circle + 100, ids.path + 100, ids.group)

    op = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE | DT_MASKS_STATE_INTERSECTION
    members = (
        pack_group_member_v9(circleA_id, group_id, op)
        + pack_group_member_v9(squareA_id, group_id, op)
        + pack_group_member_v9(circleB_id, group_id, op | DT_MASKS_STATE_GROUP_BREAK)
        + pack_group_member_v9(squareB_id, group_id, op)
    )

    masks_rows = [
        (circleA_id, DT_MASKS_CIRCLE, "circle #1",
         pack_circle(CIRCLE_CX, CIRCLE_CY, CIRCLE_R, CIRCLE_BORDER).hex(), 1),
        (squareA_id, DT_MASKS_PATH, "square #1",
         pack_path(SQUARE_CORNERS).hex(), len(SQUARE_CORNERS)),
        (circleB_id, DT_MASKS_CIRCLE, "circle #2",
         pack_circle(CIRCLE2_CX, CIRCLE2_CY, CIRCLE2_R, CIRCLE2_BORDER).hex(), 1),
        (squareB_id, DT_MASKS_PATH, "square #2",
         pack_path(SQUARE2_CORNERS).hex(), len(SQUARE2_CORNERS)),
        (group_id, DT_MASKS_GROUP, "grp exposure", members.hex(), 4),
    ]
    exposure_num = len(PIPELINE)
    masks_rows = [(exposure_num,) + r for r in masks_rows]

    bp = pack_blend_params(
        mask_mode=DEVELOP_MASK_MASK | DEVELOP_MASK_ENABLED,
        blend_cst=DEVELOP_BLEND_CS_RGB_SCENE,
        mask_combine=DEVELOP_COMBINE_NORM,
        mask_id=group_id,
        blendif=0,
    )

    global DEVELOP_MASKS_VERSION
    saved_version = DEVELOP_MASKS_VERSION
    DEVELOP_MASKS_VERSION = 9  # pre-group_start; exercises v9->v10 on load
    try:
        path = build_xmp("I1_two_adjacent_intersect_groups", bp, masks_rows)
    finally:
        DEVELOP_MASKS_VERSION = saved_version
    return path


BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")


def build_baselines():
    """Two reference renders used by verify_effect.sh to confirm each
    scenario's mask actually has a spatial effect (as opposed to being an
    accidental no-op): the exposure module fully disabled (a scenario that
    collapses to this is a hard "always zero" mask), and the module fully
    enabled with mask_mode = DEVELOP_MASK_ENABLED / no MASK or CONDITIONAL
    bits (uniform full-frame effect -- a scenario that collapses to this is
    a hard "always opaque" mask, not a real shaped/curve one)."""
    os.makedirs(BASELINE_DIR, exist_ok=True)

    build_xmp("ZBASE_module_off", pack_blend_params(), [],
               outdir=BASELINE_DIR, exposure_enabled=False)

    bp_full = pack_blend_params(
        mask_mode=DEVELOP_MASK_ENABLED,
        blend_cst=DEVELOP_BLEND_CS_RGB_SCENE,
        mask_combine=DEVELOP_COMBINE_NORM,
        mask_id=0,
        blendif=0,
    )
    build_xmp("ZBASE_mask_disabled", bp_full, [], outdir=BASELINE_DIR)


def main():
    base_id = 100000
    generated = []
    for i, sc in enumerate(SCENARIOS):
        ids = MaskIds(base_id + i * 10)
        masks_rows = []
        mask_id = sc["mask_id"]
        if sc["draw"] is not None:
            _, circle_inv, square_op, square_inv = sc["draw"]
            circle_state = DT_MASKS_STATE_SHOW | DT_MASKS_STATE_USE
            if circle_inv:
                circle_state |= DT_MASKS_STATE_INVERSE
            square_state = op_state(square_op, square_inv)
            exposure_num = len(PIPELINE)
            masks_rows = masks_history_rows(exposure_num, ids, circle_state, square_state,
                                            circle_opacity=sc["circle_opacity"],
                                            circle_refine=sc["circle_refine"])
            mask_id = ids.group

        bp = pack_blend_params(
            mask_mode=sc["mask_mode"] | DEVELOP_MASK_ENABLED,
            blend_cst=DEVELOP_BLEND_CS_RGB_SCENE,
            mask_combine=sc["mask_combine"],
            mask_id=mask_id,
            blendif=sc["blendif"],
            blendif_parameters=sc["blendif_parameters"],
        )
        path = build_xmp(sc["name"], bp, masks_rows)
        generated.append(sc["name"])
        print(f"wrote {path}")

    path = build_group_start_scenario()
    generated.append("I1_two_adjacent_intersect_groups")
    print(f"wrote {path}")

    print(f"\n{len(generated)} scenarios generated: {', '.join(generated)}")

    build_baselines()
    print(f"baselines written to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
