#!/usr/bin/env python3
#
# Extracts the user-visible strings of data/masks_group_presets.json (preset
# names and descriptions, group names and every page of their notes) as
# _("...") calls, so that xgettext finds them. The output is not compiled:
# darktable reads the JSON at runtime and translates each string with _(),
# like styles_string.h.
#
# usage: generate_masks_presets_strings.py <masks_group_presets.json> <output.h>

import json
import sys


def group_strings(group):
    if group.get("name"):
        yield group["name"]
    yield from group.get("notes", [])
    for sub in group.get("groups", []):
        yield from group_strings(sub)


def main(src, out):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    strings = []
    for preset in data.get("presets", []):
        for key in ("name", "description"):
            if preset.get(key):
                strings.append(preset[key])
        strings.extend(group_strings(preset.get("mask", {})))

    with open(out, "w", encoding="utf-8") as f:
        f.write("// Not to be compiled, generated for translation only\n\n")
        # json.dumps escapes quotes, backslashes and newlines the way C does
        for s in sorted(set(strings)):
            f.write("_(%s)\n" % json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
