#!/bin/bash
#
# Regression suite for the classic -> flexi mask migration
# (src/develop/masks/migrate_legacy.c, entry point
# dt_masks_migrate_classic_to_flexi).
#
# The XMPs under xmps/ are hand-packed classic (pre-flexi) blend_params /
# mask-point blobs -- see gen_xmp.py -- covering drawn-only, parametric-only
# and drawn+parametric masks crossed with the classic combine operators
# (union/intersection/difference/exclusion), per-shape and group-level
# polarity invert (INV/MASKS_POS), and DEVELOP_COMBINE_INCL. Rendering each
# one with darktable-cli forces the migration to run (classic mode is a
# legacy-params-only code path; a fresh XMP always loads as classic and gets
# migrated to DT_MASKS_PARAMETRIC/DT_MASKS_GROUP forms on load).
#
# Two modes:
#
#   ./run.sh                 - compare current darktable-cli output against
#                               the checked-in expected/*.png (exact pixel
#                               match). This is the normal regression mode.
#
#   ./run.sh --pristine <bin> - re-run the original validation methodology:
#                               render once with <bin> (a build with
#                               migrate_legacy.c's effects stashed out, i.e.
#                               true pre-migration classic rendering) and
#                               once with the current DARKTABLE_CLI/PATH
#                               binary, and diff the two directly. Use this
#                               only when re-validating the migration itself
#                               against a from-scratch pristine build; not
#                               needed for ordinary regression testing.
#
# See masks_revamp_flexi_migration_benchmark.md (in this directory) for the
# full investigation history (bugs found, formulas derived) behind this
# matrix.

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

PRISTINE_BIN=""
if [[ ${1:-} == --pristine ]]; then
    PRISTINE_BIN=$2
    [[ -z $PRISTINE_BIN ]] && echo "--pristine requires a binary path" >&2 && exit 1
fi

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
    local bin=$1 xmp=$2 out=$3
    "$bin" --width 400 --height 225 --hq true --apply-custom-presets false \
        "$IMAGE" "$xmp" "$out" "${CORE_OPTIONS[@]}" \
        > "${out%.png}.log" 2>&1
}

FAIL=0
COUNT=0

for xmp in xmps/*.xmp; do
    name=$(basename "$xmp" .xmp)
    COUNT=$((COUNT + 1))
    rm -f "$OUTDIR/$name"*.png

    if [[ -n $PRISTINE_BIN ]]; then
        render "$PRISTINE_BIN" "$xmp" "$OUTDIR/${name}_pristine.png"
        render "$CLI" "$xmp" "$OUTDIR/${name}_migrated.png"
        ref="$OUTDIR/${name}_pristine.png"
        got="$OUTDIR/${name}_migrated.png"
    else
        render "$CLI" "$xmp" "$OUTDIR/${name}.png"
        ref="expected/${name}.png"
        got="$OUTDIR/${name}.png"
    fi

    if [[ ! -f $got ]]; then
        echo "FAIL $name (render failed, see $OUTDIR/${name}*.log)"
        FAIL=$((FAIL + 1))
        continue
    fi
    if [[ ! -f $ref ]]; then
        echo "FAIL $name (missing $ref)"
        FAIL=$((FAIL + 1))
        continue
    fi

    diff=$(python3 "$COUNT_DIFF" "$ref" "$got")
    if [[ $diff == 0 ]]; then
        echo "OK   $name"
    else
        echo "FAIL $name ($diff differing pixels)"
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "$((COUNT - FAIL)) / $COUNT scenarios OK"
[[ $FAIL == 0 ]] && exit 0 || exit 1
