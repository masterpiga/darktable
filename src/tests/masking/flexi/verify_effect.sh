#!/bin/bash
#
# Sanity check that every scenario's mask actually does something.
#
# Each scenario applies a +5EV exposure bump through a mask; this renders
# every scenario plus two references -- baselines/ZBASE_module_off.xmp
# (exposure module disabled: the "always zero" reference) and
# baselines/ZBASE_mask_disabled.xmp (module on, mask uniform/full-frame:
# the "always opaque" reference) -- and classifies each scenario by exact
# pixel-diff against both:
#
#   PARTIAL  - differs from both references -> a real, spatially-varying
#              mask. Expected for every drawn/parametric/combined scenario
#              (A/B/C series).
#   CONSTANT_ZERO / CONSTANT_FULL - bit-identical to one of the two
#              references -> the mask has no shape at all, just a flat
#              opacity. Expected *only* for the D/E series, which exist
#              specifically to probe DEVELOP_COMBINE_INCL's constant-
#              collapse behavior (see the comments above each D/E
#              scenario in gen_xmp.py and the "Round 3" section of
#              masks_revamp_flexi_migration_benchmark.md).
#
# Any A/B/C scenario that comes back CONSTANT, or any D/E scenario that
# comes back PARTIAL or the wrong constant, means either the test fixture
# or the migration itself regressed -- this script exits non-zero in that
# case.

set -u
CDPATH=
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"

if [[ -z ${DARKTABLE_CLI:-} ]] && [[ -z $(command -v darktable-cli) ]]; then
    echo "Make sure darktable-cli is in the PATH, or set DARKTABLE_CLI" >&2
    exit 1
fi
CLI=${DARKTABLE_CLI:-darktable-cli}
COUNT_DIFF=../../integration/count-diff-pixels
IMAGE="$HERE/images/Sweep_sRGB_Linear_Half_Zip_01.tif"

# name -> expected classification (PARTIAL / CONSTANT_ZERO / CONSTANT_FULL).
# Plain case statement, not an associative array: the darktable pre-commit
# hook and some contributors' /bin/bash are still bash 3.2 (no `declare -A`).
expected_classification()
{
    case "$1" in
        A1_union|A2_intersection|A3_difference|A4_exclusion| \
        A5_union_circle_inverted|A6_intersection_group_invert| \
        B1_1channel|B2_2channel|B3_3channel| \
        B4_1channel_inverted_polarity|B5_2channel_group_invert| \
        C1_union_2channel|C2_intersection_1channel_inverted| \
        C3_difference_drawn_inverted_3channel| \
        C4_union_1channel_group_invert| \
        F1_content_incl_allchannels_noinv|F2_content_incl_allchannels_inv| \
        H1_drawn_opacity_refinement|H2_combined_opacity_refinement| \
        I1_two_adjacent_intersect_groups)
            echo PARTIAL ;;
        # J: refinement at each scope. Refinement reshapes an already-shaped
        # mask, so every one of these must stay spatially varying -- a J case
        # that collapses to a constant means the refinement swallowed the mask
        # rather than refining it.
        J1_refine_global|J2_refine_element_head|J3_refine_element_tail|\
        J4_refine_element_both|J5_refine_group|J6_refine_group_and_global|\
        J7_refine_group_of_two|J8_refine_global_of_two)
            echo PARTIAL ;;
        D1_maskspos_no_drawn_content|E2_pure_incl_and_inv)
            echo CONSTANT_ZERO ;;
        D2_parametric_incl|E1_pure_incl_only|E3_content_incl_only| \
        E4_content_incl_maskspos|E5_nocontent_maskspos_and_incl| \
        E6_nocontent_incl_only_opaque|G1_bare_uniform)
            echo CONSTANT_FULL ;;
        *)
            echo "" ;;
    esac
}

OUTDIR=$(mktemp -d)
trap 'rm -rf "$OUTDIR"' EXIT

CORE_OPTIONS=(--core --disable-opencl --library :memory:
    --conf host_memory_limit=8192
    --conf resourcelevel=reference
    --conf worker_threads=4 -t 4
    --conf plugins/lighttable/export/pixel_interpolator=lanczos3
    --conf plugins/lighttable/export/pixel_interpolator_warp=bicubic
    --conf plugins/lighttable/export/iccintent=0)

render()
{
    local xmp=$1 out=$2
    "$CLI" --width 400 --height 225 --hq true --apply-custom-presets false \
        "$IMAGE" "$xmp" "$out" "${CORE_OPTIONS[@]}" \
        > "${out%.png}.log" 2>&1
}

render baselines/ZBASE_module_off.xmp "$OUTDIR/ZBASE_module_off.png"
render baselines/ZBASE_mask_disabled.xmp "$OUTDIR/ZBASE_mask_disabled.png"

FAIL=0
for xmp in xmps/*.xmp; do
    name=$(basename "$xmp" .xmp)
    render "$xmp" "$OUTDIR/${name}.png"

    d_zero=$(python3 "$COUNT_DIFF" "$OUTDIR/ZBASE_module_off.png" "$OUTDIR/${name}.png")
    d_full=$(python3 "$COUNT_DIFF" "$OUTDIR/ZBASE_mask_disabled.png" "$OUTDIR/${name}.png")

    if [[ $d_zero == 0 ]]; then
        got=CONSTANT_ZERO
    elif [[ $d_full == 0 ]]; then
        got=CONSTANT_FULL
    else
        got=PARTIAL
    fi

    exp=$(expected_classification "$name")
    if [[ -z $exp ]]; then
        echo "FAIL $name (no expected classification registered)"
        FAIL=$((FAIL + 1))
    elif [[ $got == "$exp" ]]; then
        echo "OK   $name -> $got"
    else
        echo "FAIL $name -> $got (expected $exp; vs-off=$d_zero vs-full=$d_full)"
        FAIL=$((FAIL + 1))
    fi
done

echo
[[ $FAIL == 0 ]] && echo "all scenarios classified as expected" || echo "$FAIL scenario(s) misclassified"
exit $((FAIL != 0))
