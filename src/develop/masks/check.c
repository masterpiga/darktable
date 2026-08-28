/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "develop/masks/check.h"

#include "config.h"
#include "develop/masks/roundtrip.h"
#include "develop/masks/styleapply.h"
#include "develop/masks/verify.h"

#include <glib/gstdio.h>
#include <stdio.h>

gboolean dt_masks_check_harvest(const char *json_path, const char *report_path)
{
  setvbuf(stdout, NULL, _IOLBF, 0);

  FILE *rf = report_path ? g_fopen(report_path, "wb") : NULL;
  if(report_path && !rf)
    fprintf(stderr, "[check] cannot write %s -- continuing without a report\n",
            report_path);

  if(rf)
    fprintf(rf, "{\n  \"source\": \"%s\",\n  \"darktable_version\": \"%s\",\n",
            json_path, darktable_package_version);

  // All three run unconditionally. A contributor's harvest may be the only one
  // we ever get from that library, so stopping at the first failure would throw
  // away the other two answers about the same corpus.
  if(rf) fputs("  \"roundtrip\": {", rf);
  const gboolean rt = dt_masks_roundtrip_harvest_section(json_path, rf);
  if(rf) fputs("\n  },\n", rf);

  if(rf) fputs("  \"verify\": {", rf);
  const gboolean vf = dt_masks_verify_harvest_section(json_path, rf);
  if(rf) fputs("\n  },\n", rf);

  if(rf) fputs("  \"styleapply\": {", rf);
  gboolean sa_ran = TRUE;
  const gboolean sa = dt_masks_styleapply_harvest_section(json_path, rf, &sa_ran);
  if(rf) fputs("\n  },\n", rf);

  const gboolean passed = rt && vf && sa;

  if(rf)
  {
    fputs("  \"summary\": {\n", rf);
    fprintf(rf, "    \"passed\": %s,\n", passed ? "true" : "false");
    fprintf(rf, "    \"roundtrip_passed\": %s,\n", rt ? "true" : "false");
    fprintf(rf, "    \"verify_passed\": %s,\n", vf ? "true" : "false");
    fprintf(rf, "    \"styleapply_ran\": %s,\n", sa_ran ? "true" : "false");
    fprintf(rf, "    \"styleapply_passed\": %s\n", sa ? "true" : "false");
    fputs("  }\n}\n", rf);
    fclose(rf);
  }

  printf("\n[check] ==========================================================\n");
  printf("[check] round-trip  : %s\n", rt ? "passed" : "FAILED");
  printf("[check] verify      : %s\n", vf ? "passed" : "FAILED");
  printf("[check] style-apply : %s\n",
         !sa_ran ? "not applicable (no drawn-mask edit to host a style)"
                 : (sa ? "passed" : "FAILED"));
  printf("[check] overall     : %s\n", passed ? "PASSED" : "FAILED");
  if(report_path)
    printf("[check] combined report written to %s\n", report_path);
  printf("[check] ==========================================================\n");

  return passed;
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
