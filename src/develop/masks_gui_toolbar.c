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

// The flexi masks panel's toolbar: two runs of add buttons and a presets
// button, on one line when the panel is wide enough and on two otherwise.
//
//   one line:  [    run A | gap | run B    ] gap [presets]
//   two rows:  [    run A    ] gap [presets]
//              [    run B    ]
//
// The runs are centered on the full width, pushed left only as far as they
// must be to clear the presets button, which keeps to the right edge.
//
// This is a height-for-width container rather than a box whose children are
// moved around from "size-allocate": the row count is decided inside GTK's
// own measure and allocate passes, so nothing is reparented, shown, hidden or
// resized while GTK is laying out (see masks_toolbar in blend.h for the
// schemes that were tried that way and dropped).

#include "develop/blend_gui_internal.h"

typedef enum
{
  _TB_RUN_A = 0,
  _TB_RUN_B,
  _TB_PRESETS,
  _TB_GAP, // never drawn, only measured: its CSS width is the spacing
  _TB_N
} _tb_slot_t;

typedef struct
{
  GtkContainer parent;
  GtkWidget *child[_TB_N];
} DtMasksToolbar;

typedef struct
{
  GtkContainerClass parent_class;
} DtMasksToolbarClass;

G_DEFINE_TYPE(DtMasksToolbar, _masks_toolbar, GTK_TYPE_CONTAINER)

#define _TB(w) ((DtMasksToolbar *)(w))

// natural size of a slot, zero when it is empty or hidden
typedef struct
{
  int w[_TB_N], h[_TB_N];
} _tb_sizes_t;

static void _tb_measure(DtMasksToolbar *tb, _tb_sizes_t *s)
{
  for(int i = 0; i < _TB_N; i++)
  {
    GtkWidget *c = tb->child[i];
    s->w[i] = s->h[i] = 0;
    if(!c || !gtk_widget_get_visible(c)) continue;
    gtk_widget_get_preferred_width(c, NULL, &s->w[i]);
    gtk_widget_get_preferred_height(c, NULL, &s->h[i]);
  }
}

static inline int _tb_gap_after(const _tb_sizes_t *s, const int a, const int b)
{
  return s->w[a] && s->w[b] ? s->w[_TB_GAP] : 0;
}

// the width of the whole toolbar on one line
static int _tb_one_line_width(const _tb_sizes_t *s)
{
  const int runs = s->w[_TB_RUN_A] + _tb_gap_after(s, _TB_RUN_A, _TB_RUN_B) + s->w[_TB_RUN_B];
  return runs + (runs && s->w[_TB_PRESETS] ? s->w[_TB_GAP] : 0) + s->w[_TB_PRESETS];
}

// the narrowest the two-row layout gets
static int _tb_two_row_width(const _tb_sizes_t *s)
{
  const int row1 = s->w[_TB_RUN_A] + _tb_gap_after(s, _TB_RUN_A, _TB_PRESETS)
                   + s->w[_TB_PRESETS];
  return MAX(row1, s->w[_TB_RUN_B]);
}

static gboolean _tb_fits_one_line(const _tb_sizes_t *s, const int width)
{
  return !s->w[_TB_RUN_B] || width >= _tb_one_line_width(s);
}

static int _tb_height(const _tb_sizes_t *s, const gboolean one_line)
{
  const int row1 = MAX(s->h[_TB_RUN_A], s->h[_TB_PRESETS]);
  return one_line ? MAX(row1, s->h[_TB_RUN_B]) : row1 + s->h[_TB_RUN_B];
}

static GtkSizeRequestMode _tb_get_request_mode(GtkWidget *widget)
{
  return GTK_SIZE_REQUEST_HEIGHT_FOR_WIDTH;
}

static void _tb_get_preferred_width(GtkWidget *widget, int *minimum, int *natural)
{
  _tb_sizes_t s;
  _tb_measure(_TB(widget), &s);
  *minimum = _tb_two_row_width(&s);
  *natural = MAX(*minimum, _tb_one_line_width(&s));
}

static void _tb_get_preferred_height_for_width(GtkWidget *widget,
                                               const int width,
                                               int *minimum,
                                               int *natural)
{
  _tb_sizes_t s;
  _tb_measure(_TB(widget), &s);
  *minimum = *natural = _tb_height(&s, _tb_fits_one_line(&s, width));
}

// without a width, the height at the minimum width, as GTK expects of a
// height-for-width widget
static void _tb_get_preferred_height(GtkWidget *widget, int *minimum, int *natural)
{
  _tb_sizes_t s;
  _tb_measure(_TB(widget), &s);
  *minimum = *natural = _tb_height(&s, !s.w[_TB_RUN_B]);
}

static void _tb_get_preferred_width_for_height(GtkWidget *widget,
                                               const int height,
                                               int *minimum,
                                               int *natural)
{
  _tb_get_preferred_width(widget, minimum, natural);
}

static void _tb_place(GtkWidget *c,
                      const GtkAllocation *a,
                      const gboolean rtl,
                      const int x,
                      const int y,
                      const int w,
                      const int h)
{
  if(!c || !gtk_widget_get_visible(c)) return;
  GtkAllocation ca = { .x = a->x + (rtl ? a->width - x - w : x), .y = a->y + y,
                       .width = w, .height = h };
  gtk_widget_size_allocate(c, &ca);
}

// a run of width w, centered on the full width but kept clear of `limit`
static int _tb_center(const int width, const int w, const int limit)
{
  return MAX(0, MIN((width - w) / 2, limit - w));
}

static void _tb_size_allocate(GtkWidget *widget, GtkAllocation *a)
{
  DtMasksToolbar *tb = _TB(widget);
  gtk_widget_set_allocation(widget, a);

  _tb_sizes_t s;
  _tb_measure(tb, &s);
  const gboolean rtl = gtk_widget_get_direction(widget) == GTK_TEXT_DIR_RTL;
  const int W = a->width;
  const int gap = s.w[_TB_GAP];
  const int wa = s.w[_TB_RUN_A], wb = s.w[_TB_RUN_B], wp = s.w[_TB_PRESETS];
  const int wp_x = W - wp;
  const int limit = wp ? wp_x - gap : W;

  const int row1_h = MAX(s.h[_TB_RUN_A], s.h[_TB_PRESETS]);
  int gap_x;
  if(_tb_fits_one_line(&s, W))
  {
    const int h = _tb_height(&s, TRUE);
    const int ab_gap = _tb_gap_after(&s, _TB_RUN_A, _TB_RUN_B);
    const int x = _tb_center(W, wa + ab_gap + wb, limit);
    _tb_place(tb->child[_TB_RUN_A], a, rtl, x, 0, wa, h);
    _tb_place(tb->child[_TB_RUN_B], a, rtl, x + wa + ab_gap, 0, wb, h);
    _tb_place(tb->child[_TB_PRESETS], a, rtl, wp_x, 0, wp, h);
    gap_x = x + wa;
  }
  else
  {
    const int xa = _tb_center(W, wa, limit);
    _tb_place(tb->child[_TB_RUN_A], a, rtl, xa, 0, wa, row1_h);
    _tb_place(tb->child[_TB_PRESETS], a, rtl, wp_x, 0, wp, row1_h);
    _tb_place(tb->child[_TB_RUN_B], a, rtl, _tb_center(W, wb, W), row1_h, wb,
              s.h[_TB_RUN_B]);
    gap_x = xa + wa;
  }
  // every visible child gets an allocation; the gap's is just empty space
  _tb_place(tb->child[_TB_GAP], a, rtl, MIN(gap_x, MAX(0, W - gap)), 0, gap, row1_h);
}

static void _tb_forall(GtkContainer *container,
                       const gboolean include_internals,
                       GtkCallback callback,
                       gpointer data)
{
  DtMasksToolbar *tb = _TB(container);
  // the callback may remove the child (destroy does), so read each slot afresh
  for(int i = 0; i < _TB_N; i++)
    if(tb->child[i]) callback(tb->child[i], data);
}

static void _tb_remove(GtkContainer *container, GtkWidget *child)
{
  DtMasksToolbar *tb = _TB(container);
  for(int i = 0; i < _TB_N; i++)
  {
    if(tb->child[i] != child) continue;
    const gboolean was_visible = gtk_widget_get_visible(child);
    gtk_widget_unparent(child);
    tb->child[i] = NULL;
    if(was_visible) gtk_widget_queue_resize(GTK_WIDGET(container));
    return;
  }
}

static GType _tb_child_type(GtkContainer *container)
{
  return GTK_TYPE_WIDGET;
}

static void _masks_toolbar_class_init(DtMasksToolbarClass *klass)
{
  GtkWidgetClass *wclass = GTK_WIDGET_CLASS(klass);
  GtkContainerClass *cclass = GTK_CONTAINER_CLASS(klass);

  wclass->get_request_mode = _tb_get_request_mode;
  wclass->get_preferred_width = _tb_get_preferred_width;
  wclass->get_preferred_height = _tb_get_preferred_height;
  wclass->get_preferred_height_for_width = _tb_get_preferred_height_for_width;
  wclass->get_preferred_width_for_height = _tb_get_preferred_width_for_height;
  wclass->size_allocate = _tb_size_allocate;

  cclass->forall = _tb_forall;
  cclass->remove = _tb_remove;
  cclass->child_type = _tb_child_type;

  // styled like the box it replaced
  gtk_widget_class_set_css_name(wclass, "box");
}

static void _masks_toolbar_init(DtMasksToolbar *tb)
{
  gtk_widget_set_has_window(GTK_WIDGET(tb), FALSE);
}

GtkWidget *_masks_toolbar_new(GtkWidget *run_a,
                              GtkWidget *run_b,
                              GtkWidget *presets,
                              GtkWidget *gap)
{
  DtMasksToolbar *tb = g_object_new(_masks_toolbar_get_type(), NULL);
  GtkWidget *children[_TB_N] = { run_a, run_b, presets, gap };
  for(int i = 0; i < _TB_N; i++)
  {
    tb->child[i] = children[i];
    gtk_widget_set_parent(children[i], GTK_WIDGET(tb));
  }
  return GTK_WIDGET(tb);
}

// modelines: These editor modelines have been set for all relevant files
// by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on;
// indent-mode cstyle; remove-trailing-spaces modified;
