/*
    This file is part of darktable,
    Copyright (C) 2010-2026 darktable developers.

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
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "common/darktable.h"
#include "common/math.h"
#include "develop/develop.h"
#include "gradientslider.h"
#include "gui/gtk.h"
#include "gui/accelerators.h"

#define DTGTK_GRADIENT_SLIDER_VALUE_CHANGED_DELAY_MAX 50
#define DTGTK_GRADIENT_SLIDER_VALUE_CHANGED_DELAY_MIN 10
#define DTGTK_GRADIENT_SLIDER_DEFAULT_INCREMENT 0.01

// define GTypes
G_DEFINE_TYPE(GtkDarktableGradientSlider, _gradient_slider, GTK_TYPE_DRAWING_AREA);
#define parent_class _gradient_slider_parent_class

// Class overrides
static void _gradient_slider_get_preferred_height(GtkWidget *widget,
                                                  gint *min_height,
                                                  gint *nat_height);
static void _gradient_slider_get_preferred_width(GtkWidget *widget,
                                                 gint *min_width,
                                                 gint *nat_width);
static gboolean _gradient_slider_draw(GtkWidget *widget, cairo_t *cr);
static void _gradient_slider_dispose(GObject *object);

// Events
static void _gradient_slider_enter(GtkEventControllerMotion *controller,
                                  gdouble x,
                                  gdouble y,
                                  gpointer user_data);
static void _gradient_slider_leave(GtkEventControllerMotion *controller,
                                   gpointer user_data);
static void _gradient_slider_button_pressed(GtkGestureSingle *gesture,
                                            gint n_press,
                                            gdouble x,
                                            gdouble y,
                                            gpointer user_data);
static void _gradient_slider_button_released(GtkGestureSingle *gesture,
                                             gint n_press,
                                             gdouble x,
                                             gdouble y,
                                             gpointer user_data);
static void _gradient_slider_motion(GtkEventControllerMotion *controller,
                                    gdouble x,
                                    gdouble y,
                                    gpointer user_data);
static void _gradient_slider_scroll(GtkEventControllerScroll *controller,
                                    gdouble dx,
                                    gdouble dy,
                                    gpointer user_data);
static gboolean _gradient_slider_key_pressed(GtkEventControllerKey *controller,
                                             guint keyval,
                                             guint keycode,
                                             GdkModifierType state,
                                             gpointer user_data);

enum
{
  VALUE_CHANGED,
  VALUE_RESET,
  LAST_SIGNAL
};

static guint _signals[LAST_SIGNAL] = { 0 };

static gboolean _gradient_slider_postponed_value_change(gpointer data)
{
  if(!GTK_IS_WIDGET(data)) return 0;

  if(DTGTK_GRADIENT_SLIDER(data)->is_changed == TRUE)
  {
    g_signal_emit_by_name(G_OBJECT(data), "value-changed");
    DTGTK_GRADIENT_SLIDER(data)->is_changed = FALSE;
  }

  if(!DTGTK_GRADIENT_SLIDER(data)->is_dragging) DTGTK_GRADIENT_SLIDER(data)->timeout_handle = 0;
  else
  {
    const int delay = CLAMP(darktable.develop->full.pipe->average_delay * 3 / 2,
                            DTGTK_GRADIENT_SLIDER_VALUE_CHANGED_DELAY_MIN,
                            DTGTK_GRADIENT_SLIDER_VALUE_CHANGED_DELAY_MAX);
    DTGTK_GRADIENT_SLIDER(data)->timeout_handle = g_timeout_add(delay, _gradient_slider_postponed_value_change, data);
  }

  return FALSE; // This is called by the gtk mainloop and is threadsafe
}

static inline gboolean _test_if_marker_is_upper_or_down(const gint marker,
                                                        const gboolean up)
{
  if(up && (marker == GRADIENT_SLIDER_MARKER_LOWER_OPEN ||
                  marker == GRADIENT_SLIDER_MARKER_LOWER_FILLED ||
                  marker == GRADIENT_SLIDER_MARKER_LOWER_OPEN_BIG ||
                  marker == GRADIENT_SLIDER_MARKER_LOWER_FILLED_BIG))
    return FALSE;
  else if(!up && (marker == GRADIENT_SLIDER_MARKER_UPPER_OPEN ||
                  marker == GRADIENT_SLIDER_MARKER_UPPER_FILLED ||
                  marker == GRADIENT_SLIDER_MARKER_UPPER_OPEN_BIG ||
                  marker == GRADIENT_SLIDER_MARKER_UPPER_FILLED_BIG))
    return FALSE;
  else
    return TRUE; // must be a DOUBLE
}

static inline gdouble _screen_to_scale(GtkWidget *widget,
                                       const gint screen)
{
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  return ((gdouble)screen - gslider->margin_left) / ((gdouble)allocation.width - gslider->margin_left - gslider->margin_right);
}

static inline gint _scale_to_screen(GtkWidget *widget,
                                    const gdouble scale)
{
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  return (gint)(scale * (allocation.width - gslider->margin_left - gslider->margin_right) + gslider->margin_left);
}

static inline gdouble _get_position_from_screen(GtkWidget *widget,
                                                const gdouble x)
{
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  gdouble position = roundf(_screen_to_scale(widget, x) / gslider->increment) * gslider->increment;
  return CLAMP(position, 0., 1.);
}

static inline gint _get_active_marker(GtkDarktableGradientSlider *gslider)
{
  return (gslider->selected >= 0) ? gslider->selected : gslider->active;
}

static inline void _clamp_marker(GtkDarktableGradientSlider *gslider,
                                 const gint selected)
{
  g_return_if_fail(gslider != NULL);

  const gdouble min = (selected == 0) ? 0.0f : gslider->position[selected - 1];
  const gdouble max = (selected == gslider->positions - 1) ? 1.0f : gslider->position[selected + 1];
  gslider->position[selected] = CLAMP(gslider->position[selected], min, max);
}

static gint _get_active_marker_from_screen(GtkWidget *widget,
                                           const gdouble x,
                                           const gdouble y)
{
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  assert(gslider->positions > 0);

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  const gboolean up = (y <= allocation.height / 2.f);

  // side-aware selection: from the vertical middle of the slider and above,
  // only markers drawn on the upper half are eligible, and likewise below --
  // this keeps close-together upper/lower markers (e.g. a zero-width
  // feather/range band) independently selectable based on which side the
  // pointer is on. DOUBLE markers (drawn full height) match either side and
  // remain always eligible; if for some reason no marker matches the
  // pointer's side, fall back to the closest marker overall.
  const gdouble newposition = _get_position_from_screen(widget, x);

  gint best = -1;
  gdouble bestdx = 0;
  gint best_any = 0;
  gdouble bestdx_any = fabs(newposition - gslider->position[0]);

  for(int k = 0; k < gslider->positions; k++)
  {
    const gdouble dx = fabs(newposition - gslider->position[k]);
    if(dx < bestdx_any - 1e-9)
    {
      bestdx_any = dx;
      best_any = k;
    }

    if(_test_if_marker_is_upper_or_down(gslider->marker[k], up)
       && (best == -1 || dx < bestdx - 1e-9))
    {
      best = k;
      bestdx = dx;
    }
  }

  return (best >= 0) ? best : best_any;
}

static gdouble _slider_move(GtkWidget *widget,
                            const gint k,
                            const gdouble value,
                            const gint direction)
{
  g_return_val_if_fail(DTGTK_IS_GRADIENT_SLIDER(widget), value);

  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);

  gdouble newvalue = value;
  gdouble leftnext, rightnext, ms;

  switch(gslider->markers_type)
  {
    case FREE_MARKERS:
    {
      leftnext = (k == 0) ? 0.0f : gslider->position[k - 1];
      rightnext = (k == gslider->positions - 1) ? 1.0f : gslider->position[k + 1];
      ms = gslider->min_spacing;
      switch(direction)
      {
        case MOVE_LEFT:
          if(value < leftnext + ms)
            newvalue = (k == 0) ? fmax(value, 0.0f) : _slider_move(widget, k - 1, value - ms, direction) + ms;
          break;
        case MOVE_RIGHT:
          if(value > rightnext - ms)
            newvalue = (k == gslider->positions - 1) ? fmin(value, 1.0f) : _slider_move(widget, k + 1, value +  ms, direction) - ms;
          break;
      }
      break;
    }
    case PROPORTIONAL_MARKERS:
    {
      ms = fmax(gslider->min_spacing, 1.0e-6);
      const double vmin = ((k == 0) ? 0.0f : gslider->position[0]);
      const double vmax = ((k == gslider->positions - 1) ? 1.0f : gslider->position[gslider->positions - 1]);

      newvalue = CLAMP(value, vmin + ms * k, vmax - ms * (gslider->positions - 1 - k));
      const double rl = (newvalue - gslider->position[0]) / (gslider->position[k] - gslider->position[0]);
      const double rh = (gslider->position[gslider->positions - 1] - newvalue) /
                        (gslider->position[gslider->positions - 1] - gslider->position[k]);

      for(int i = 1; i < k; i++)
        gslider->position[i] = rl * (gslider->position[i] - gslider->position[0]) + gslider->position[0];

      for(int i = k + 1; i < gslider->positions; i++)
        gslider->position[i] = gslider->position[gslider->positions - 1] -
                               rh * (gslider->position[gslider->positions - 1] - gslider->position[i]);
      break;
    }
  }
  gslider->position[k] = newvalue;
  return newvalue;
}

static gboolean _gradient_slider_add_delta_internal(GtkWidget *widget,
                                                    gdouble delta,
                                                    const guint state,
                                                    const gint selected)
{
  g_return_val_if_fail(DTGTK_IS_GRADIENT_SLIDER(widget), TRUE);

  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);

  if(selected == -1) return TRUE;

  delta *= dt_accel_get_speed_multiplier(widget, state);

  gslider->position[selected] = gslider->position[selected] + delta;
  _clamp_marker(gslider, selected);

  gtk_widget_queue_draw(widget);
  g_signal_emit_by_name(G_OBJECT(widget), "value-changed");

  return TRUE;
}

static float _default_linear_scale_callback(GtkWidget *self,
                                            const float value,
                                            const int dir)
{
  // regardless of dir: input <-> output
  return value;
}

static void _gradient_slider_enter(GtkEventControllerMotion *controller,
                                   gdouble x,
                                   gdouble y,
                                   gpointer user_data)
{
  GtkWidget *widget = dt_gui_get_widget(controller);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  gtk_widget_set_state_flags(widget, GTK_STATE_FLAG_PRELIGHT, TRUE);
  gslider->is_entered = TRUE;
  gtk_widget_queue_draw(widget);
}

static void _gradient_slider_leave(GtkEventControllerMotion *controller,
                                   gpointer user_data)
{
  GtkWidget *widget = dt_gui_get_widget(controller);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  if(!(gslider->is_dragging))
  {
    gtk_widget_set_state_flags(widget, GTK_STATE_FLAG_NORMAL, TRUE);
    gslider->is_entered = FALSE;
    gslider->active = -1;
    gtk_widget_queue_draw(widget);
  }
}

static void _gradient_slider_button_pressed(GtkGestureSingle *gesture,
                                            gint n_press,
                                            gdouble x,
                                            gdouble y,
                                            gpointer user_data)
{
  GtkWidget *widget = dt_gui_get_widget(gesture);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  const guint button = gtk_gesture_single_get_current_button(gesture);

  // reset slider on double click
  if(button == GDK_BUTTON_PRIMARY && n_press == 2 && gslider->is_resettable)
  {
    gslider->is_dragging = FALSE;
    gslider->do_reset = TRUE;
    gslider->selected = -1;
    for(int k = 0; k < gslider->positions; k++) gslider->position[k] = gslider->resetvalue[k];
    gtk_widget_queue_draw(widget);
    g_signal_emit_by_name(G_OBJECT(widget), "value-changed");
    g_signal_emit_by_name(G_OBJECT(widget), "value-reset");
  }
  else if((button == GDK_BUTTON_PRIMARY || button == GDK_BUTTON_SECONDARY) && n_press == 1)
  {
    const gint lselected = _get_active_marker_from_screen(widget, x, y);

    assert(lselected >= 0);
    assert(lselected <= gslider->positions - 1);

    if(button == GDK_BUTTON_PRIMARY) // left mouse button : select and start dragging
    {
      gslider->selected = lselected;
      gslider->do_reset = FALSE;

      const gdouble newposition = _get_position_from_screen(widget, x);
      const gint direction = gslider->position[gslider->selected] <= newposition ? MOVE_RIGHT : MOVE_LEFT;

      _slider_move(widget, gslider->selected, newposition, direction);

      gslider->is_changed = TRUE;
      gslider->is_dragging = TRUE;
      // timeout_handle should always be zero here, but check just in case
      const int delay = CLAMP(darktable.develop->full.pipe->average_delay * 3 / 2,
                              DTGTK_GRADIENT_SLIDER_VALUE_CHANGED_DELAY_MIN,
                              DTGTK_GRADIENT_SLIDER_VALUE_CHANGED_DELAY_MAX);
      if(!gslider->timeout_handle)
        gslider->timeout_handle = g_timeout_add(delay, _gradient_slider_postponed_value_change, widget);
    }
    else if(gslider->positions > 1) // right mouse button: switch on/off selection (only if we have more than one marker)
    {
      gslider->is_dragging = FALSE;
      gslider->do_reset = FALSE;

      if(gslider->selected != lselected)
        gslider->selected = lselected;
      else
        gslider->selected = -1;

      gtk_widget_queue_draw(widget);
    }
  }

  // claim the sequence so an ancestor's own gesture (e.g. the scrolled
  // panel this slider lives in) can't steal it once the pointer starts
  // moving -- without this, dragging a marker is cancelled after the
  // first move: the panel's gesture claims the in-progress sequence, and
  // this widget's own gesture receives "cancel" (see _gesture_cancel,
  // which synthesizes a "released" and ends the drag right there)
  dt_gui_claim(gesture);
}

static void _gradient_slider_motion(GtkEventControllerMotion *controller,
                                    gdouble x,
                                    gdouble y,
                                    gpointer user_data)
{
  GtkWidget *widget = dt_gui_get_widget(controller);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);

  if(gslider->is_dragging == TRUE && gslider->selected != -1 && gslider->do_reset == FALSE)
  {
    assert(gslider->timeout_handle > 0);

    const gdouble newposition = _get_position_from_screen(widget, x);
    const gint direction = gslider->position[gslider->selected] <= newposition ? MOVE_RIGHT : MOVE_LEFT;

    _slider_move(widget, gslider->selected, newposition, direction);

    gslider->is_changed = TRUE;

    gtk_widget_queue_draw(widget);
  }
  else
  {
    const gint active = _get_active_marker_from_screen(widget, x, y);
    if(active != gslider->active)
    {
      gslider->active = active;
      gtk_widget_queue_draw(widget);
    }
  }

  if(gslider->selected != -1) gtk_widget_grab_focus(widget);
}

static void _gradient_slider_button_released(GtkGestureSingle *gesture,
                                             gint n_press,
                                             gdouble x,
                                             gdouble y,
                                             gpointer user_data)
{
  GtkWidget *widget = dt_gui_get_widget(gesture);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  const gint selected = _get_active_marker(gslider);

  if(gtk_gesture_single_get_current_button(gesture) == GDK_BUTTON_PRIMARY
     && selected != -1 && gslider->do_reset == FALSE)
  {
    // First get some dimension info
    gslider->is_changed = TRUE;
    const gdouble newposition = _get_position_from_screen(widget, x);
    const gint direction = gslider->position[selected] <= newposition ? MOVE_RIGHT : MOVE_LEFT;

    _slider_move(widget, selected, newposition, direction);

    gtk_widget_queue_draw(widget);

    gslider->is_dragging = FALSE;
    if(gslider->timeout_handle) g_source_remove(gslider->timeout_handle);
    gslider->timeout_handle = 0;
    g_signal_emit_by_name(G_OBJECT(widget), "value-changed");
  }

  dt_gui_claim(gesture);
}

static void _gradient_slider_scroll(GtkEventControllerScroll *controller,
                                    gdouble dx,
                                    gdouble dy,
                                    gpointer user_data)
{
  GtkWidget *widget = dt_gui_get_widget(controller);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);
  const gint selected = _get_active_marker(gslider);
  if(selected == -1) return;

  gtk_widget_grab_focus(widget);

  // the DISCRETE proxy already accumulated smooth deltas into unit steps
  const int delta_y = (int)dy;
  if(delta_y == 0) return;

  _gradient_slider_add_delta_internal(widget, delta_y * -gslider->increment,
                                      dt_gui_get_current_event_state(GTK_EVENT_CONTROLLER(controller)),
                                      selected);
}

static gboolean _gradient_slider_key_pressed(GtkEventControllerKey *controller,
                                             guint keyval,
                                             guint keycode,
                                             GdkModifierType state,
                                             gpointer user_data)
{
  GtkWidget *widget = dt_gui_get_widget(controller);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);

  int handled = FALSE;
  float delta = -gslider->increment;
  switch(keyval)
  {
    case GDK_KEY_Up:
    case GDK_KEY_KP_Up:
    case GDK_KEY_Right:
    case GDK_KEY_KP_Right:
      delta = gslider->increment;
    case GDK_KEY_Down:
    case GDK_KEY_KP_Down:
    case GDK_KEY_Left:
    case GDK_KEY_KP_Left:
      handled = TRUE;
  }

  if(!handled) return FALSE;

  const gint selected = _get_active_marker(gslider);
  if(selected == -1) return TRUE;

  return _gradient_slider_add_delta_internal(widget, delta, state, selected);
}

static void _gradient_slider_class_init(GtkDarktableGradientSliderClass *klass)
{
  GtkWidgetClass *widget_class = (GtkWidgetClass *)klass;

  widget_class->get_preferred_height = _gradient_slider_get_preferred_height;
  widget_class->get_preferred_width = _gradient_slider_get_preferred_width;
  widget_class->draw = _gradient_slider_draw;
  GObjectClass *object_class = (GObjectClass *)klass;
  object_class->dispose = _gradient_slider_dispose;

  _signals[VALUE_CHANGED] = g_signal_new("value-changed",
                                         G_TYPE_FROM_CLASS(klass),
                                         G_SIGNAL_RUN_LAST, 0,
                                         NULL, NULL,
                                         g_cclosure_marshal_VOID__VOID,
                                         G_TYPE_NONE, 0);
  _signals[VALUE_RESET] = g_signal_new("value-reset",
                                       G_TYPE_FROM_CLASS(klass),
                                       G_SIGNAL_RUN_LAST, 0,
                                       NULL, NULL,
                                       g_cclosure_marshal_VOID__VOID,
                                       G_TYPE_NONE, 0);
}

static void _gradient_slider_init(GtkDarktableGradientSlider *gslider)
{
  g_return_if_fail(gslider != NULL);

  GtkWidget *widget = GTK_WIDGET(gslider);
  gtk_widget_add_events(widget,
                        GDK_EXPOSURE_MASK |
                        GDK_BUTTON_PRESS_MASK |
                        GDK_BUTTON_RELEASE_MASK |
                        GDK_ENTER_NOTIFY_MASK |
                        GDK_LEAVE_NOTIFY_MASK |
                        GDK_KEY_PRESS_MASK |
                        GDK_KEY_RELEASE_MASK |
                        GDK_POINTER_MOTION_MASK |
                        darktable.gui->scroll_mask);

  gtk_widget_set_has_window(widget, TRUE);
  gtk_widget_set_can_focus(widget, TRUE);

  // GTK3 class handlers (button-press-event & friends) are gone in GTK4:
  // route the input through controllers, which exist in both.  n_press
  // replaces the GDK_2BUTTON_PRESS reset check, the DISCRETE scroll proxy
  // replaces the unit-delta accumulator.
  dt_gui_connect_click(widget, _gradient_slider_button_pressed,
                       _gradient_slider_button_released, NULL);
  dt_gui_connect_motion(widget, _gradient_slider_motion,
                        _gradient_slider_enter, _gradient_slider_leave, NULL);
  dt_gui_connect_scroll(widget,
                        GTK_EVENT_CONTROLLER_SCROLL_VERTICAL
                          | GTK_EVENT_CONTROLLER_SCROLL_DISCRETE,
                        _gradient_slider_scroll, NULL);
  dt_gui_connect_key(widget, _gradient_slider_key_pressed, NULL);
}

static void _gradient_slider_get_preferred_height(GtkWidget *widget,
                                                  gint *min_height,
                                                  gint *nat_height)
{
  g_return_if_fail(widget != NULL);

  GtkStyleContext *context = gtk_widget_get_style_context(widget);
  GtkStateFlags state = gtk_widget_get_state_flags(widget);

  GtkBorder margin, border, padding;
  int css_min_height;
  gtk_style_context_get(context, state, "min-height", &css_min_height, NULL);
  gtk_style_context_get_margin(context, state, &margin);
  gtk_style_context_get_border(context, state, &border);
  gtk_style_context_get_padding(context, state, &padding);
  *min_height = *nat_height = css_min_height + padding.top + padding.bottom + border.top + border.bottom + margin.top + margin.bottom;
}

static void _gradient_slider_get_preferred_width(GtkWidget *widget,
                                                 gint *min_width,
                                                 gint *nat_width)
{
  g_return_if_fail(DTGTK_IS_GRADIENT_SLIDER(widget));

  GtkStyleContext *context = gtk_widget_get_style_context(widget);
  GtkStateFlags state = gtk_widget_get_state_flags(widget);

  GtkBorder margin, border, padding;
  int css_min_width;
  gtk_style_context_get (context, state, "min-width", &css_min_width, NULL);
  gtk_style_context_get_margin(context, state, &margin);
  gtk_style_context_get_border(context, state, &border);
  gtk_style_context_get_padding(context, state, &padding);
  *min_width = *nat_width = css_min_width + padding.left + padding.right + border.left + border.right + margin.left + margin.right;

  DTGTK_GRADIENT_SLIDER(widget)->margin_left = padding.left + border.left + margin.left;
  DTGTK_GRADIENT_SLIDER(widget)->margin_right = padding.right + border.right + margin.right;
}

static void _gradient_slider_dispose(GObject *object)
{
  g_return_if_fail(DTGTK_IS_GRADIENT_SLIDER(object));

  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(object);

  if(gslider->timeout_handle)
    g_source_remove(gslider->timeout_handle);

  gslider->timeout_handle = 0;

  if(gslider->colors)
    g_list_free_full(gslider->colors, g_free);

  gslider->colors = NULL;

  G_OBJECT_CLASS(parent_class)->dispose(object);
}

static gboolean _gradient_slider_draw(GtkWidget *widget,
                                      cairo_t *cr)
{
  g_return_val_if_fail(DTGTK_IS_GRADIENT_SLIDER(widget), FALSE);
  GtkDarktableGradientSlider *gslider = DTGTK_GRADIENT_SLIDER(widget);

  assert(gslider->position > 0);

  GtkStyleContext *context = gtk_widget_get_style_context(widget);
  GtkStateFlags state = gtk_widget_get_state_flags(widget);

  GdkRGBA color;
  gtk_style_context_get_color(context, state, &color);

  GtkAllocation allocation;
  GtkBorder margin, border, padding;
  gtk_widget_get_allocation(widget, &allocation);
  gtk_style_context_get_margin(context, state, &margin);
  gtk_style_context_get_border(context, state, &border);
  gtk_style_context_get_padding(context, state, &padding);

  // Begin cairo drawing
  // for frame and background, we remove css margin from allocation
  int startx = margin.left;
  int starty = margin.top;
  int cwidth = allocation.width - margin.left - margin.right;
  int cheight = allocation.height - margin.top - margin.bottom;
  gtk_render_background(context, cr, startx, starty, cwidth, cheight);
  gtk_render_frame(context, cr, startx, starty, cwidth, cheight);

  // then we draw the content
  startx += padding.left + border.left;
  starty += padding.top + border.top;
  cwidth -= padding.left + padding.right + border.left + border.right;
  cheight -= padding.top + padding.bottom + border.top + border.bottom;
  const int y1 = round(0.3f * cheight);
  const int gheight = cheight - 2 * y1;

  // First build the cairo gradient and then fill the gradient
  if(gslider->colors)
  {
    cairo_pattern_t *gradient = cairo_pattern_create_linear(0, 0, cwidth, 0);
    for(GList *current = gslider->colors; current; current = g_list_next(current))
    {
      _gradient_slider_stop_t *stop = (_gradient_slider_stop_t *)current->data;
      cairo_pattern_add_color_stop_rgba(gradient, stop->position, stop->color.red, stop->color.green,
                                       stop->color.blue, stop->color.alpha);
    }
    if(gradient != NULL) // Do we got a gradient, lets draw it
    {
      cairo_set_line_width(cr, 0.1);
      cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
      cairo_translate(cr, 0, starty);
      cairo_set_source(cr, gradient);
      cairo_rectangle(cr, startx, y1, cwidth, gheight);
      cairo_fill(cr);
      cairo_stroke(cr);
      cairo_pattern_destroy(gradient);
    }
  }

  // Lets draw position arrows
  cairo_set_source_rgba(cr, color.red, color.green, color.blue, 1.0);

  // do we have a picker value to draw?
  if(!dt_isnan(gslider->picker[0]))
  {
    int vx_min = _scale_to_screen(widget, CLAMP(gslider->picker[1], 0.0, 1.0));
    int vx_max = _scale_to_screen(widget, CLAMP(gslider->picker[2], 0.0, 1.0));
    int vx_avg = _scale_to_screen(widget, CLAMP(gslider->picker[0], 0.0, 1.0));

    cairo_set_source_rgba(cr, color.red, color.green, color.blue, 0.33);

    cairo_rectangle(cr, vx_min, y1, fmax((float)vx_max - vx_min, 0.0f), gheight);
    cairo_fill(cr);

    cairo_set_source_rgba(cr, color.red, color.green, color.blue, 1.0);

    cairo_move_to(cr, vx_avg, y1);
    cairo_rel_line_to(cr, 0, gheight);
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
    cairo_set_line_width(cr, 1.0);
    cairo_stroke(cr);
  }

  // A 4-point open/filled/filled/open marker set is the "range + feather"
  // convention (parametric blendif channels): paint the feather zones
  // directly on the gradient bar, white and fading from the open (feather)
  // point to full opacity at the neighbouring filled (range) point, plus a
  // plain outline around the flat, fully-selected zone between the two
  // filled points. The open marker's own up/down bit decides which edge
  // (top or bottom) each wedge's point sits on, so this follows polarity
  // (the "invert" toggle swaps that bit, see _blendop_blendif_polarity_callback)
  // instead of assuming a fixed orientation.
  if(gslider->positions == 4
     && !(gslider->marker[0] & 0x01) && (gslider->marker[1] & 0x01)
     && (gslider->marker[2] & 0x01) && !(gslider->marker[3] & 0x01))
  {
    const int x0 = _scale_to_screen(widget, gslider->position[0]);
    const int x1 = _scale_to_screen(widget, gslider->position[1]);
    const int x2 = _scale_to_screen(widget, gslider->position[2]);
    const int x3 = _scale_to_screen(widget, gslider->position[3]);
    const int top = y1;
    const int bottom = cheight - y1;

    cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.55);

    const int apex0 = (gslider->marker[0] & 0x04) ? top : bottom;
    cairo_move_to(cr, x0, apex0);
    cairo_line_to(cr, x1, top);
    cairo_line_to(cr, x1, bottom);
    cairo_close_path(cr);
    cairo_fill(cr);

    const int apex3 = (gslider->marker[3] & 0x04) ? top : bottom;
    cairo_move_to(cr, x3, apex3);
    cairo_line_to(cr, x2, top);
    cairo_line_to(cr, x2, bottom);
    cairo_close_path(cr);
    cairo_fill(cr);

    cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.8);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));
    // apex0's own up/down bit already tells us the polarity (see the wedge
    // comment above): inverted means the selected zone is everything OUTSIDE
    // [x1, x2], so the outline belongs on the two exterior bands instead of
    // the interior one. marker[0] sits on the UPPER edge (bit set, hollow
    // marker above the filled one) for an inverted/excluded range, and on
    // the LOWER edge (bit clear, hollow below filled) for the normal,
    // non-inverted/included range -- same bit apex0 above already keys off.
    if(gslider->marker[0] & 0x04)
    {
      cairo_rectangle(cr, startx + 0.5, top + 0.5, fmax(x1 - startx - 1, 0), fmax(bottom - top - 1, 0));
      cairo_stroke(cr);
      cairo_rectangle(cr, x2 + 0.5, top + 0.5, fmax(startx + cwidth - x2 - 1, 0), fmax(bottom - top - 1, 0));
      cairo_stroke(cr);
    }
    else
    {
      cairo_rectangle(cr, x1 + 0.5, top + 0.5, fmax(x2 - x1 - 1, 0), fmax(bottom - top - 1, 0));
      cairo_stroke(cr);
    }
  }

  // highlight the marker the pointer is actually closest to right now.
  // gslider->selected is sticky (stays set after release so keyboard/scroll
  // input keeps targeting the last-touched marker -- see button_release),
  // so it tracks stale state, not the cursor; while dragging, follow the
  // grabbed marker instead so it stays highlighted even if the pointer
  // drifts past a neighbour. gslider->active is recomputed on every motion
  // event via the same nearest-wins hit-test as clicking uses, so hover
  // tracks the closest point even without an exact hit.
  const gint hovered_marker = gslider->pinned >= 0 ? gslider->pinned
                             : gslider->is_dragging ? gslider->selected
                             : (gslider->is_entered ? gslider->active : -1);

  for(int k = 0; k < gslider->positions; k++)
  {
    const int vx = _scale_to_screen(widget, gslider->position[k]);
    const int mk = gslider->marker[k];
    const gboolean hovered = (k == hovered_marker);
    // small circle handles straddling the bar's edge, enlarged on hover
    // so the four control points stay easy to grab
    const double r = ((mk & (1 << 3)) ? 0.55 : 0.45) * y1 * (hovered ? 1.5 : 1.0);

    if(hovered) // highlight the active marker
      cairo_set_source_rgba(cr, color.red, color.green, color.blue, 1.0);
    else
      cairo_set_source_rgba(cr, color.red * 0.8, color.green * 0.8, color.blue * 0.8, 1.0);

    cairo_set_antialias(cr, CAIRO_ANTIALIAS_DEFAULT);
    cairo_set_line_width(cr, DT_PIXEL_APPLY_DPI(1.0));

    if(mk & 0x04) /* upper handle, sits on the bar's top edge */
    {
      cairo_arc(cr, vx, y1, r, 0, 2 * M_PI);
      if(mk & 0x01) cairo_fill(cr);
      else cairo_stroke(cr);
    }

    if(mk & 0x02) /* lower handle, sits on the bar's bottom edge */
    {
      cairo_arc(cr, vx, cheight - y1, r, 0, 2 * M_PI);
      if(mk & 0x01) cairo_fill(cr);
      else cairo_stroke(cr);
    }
  }

  return FALSE;
}

gint _list_find_by_position(gconstpointer a, gconstpointer b)
{
  _gradient_slider_stop_t *stop = (_gradient_slider_stop_t *)a;
  gfloat position = *((gfloat *)b);
  return (gint)((stop->position * 100.0) - (position * 100.0));
}

static void _gradient_slider_set_defaults(GtkDarktableGradientSlider *gslider)
{
  g_return_if_fail(gslider != NULL);

  gslider->is_dragging = gslider->is_changed =
    gslider->do_reset = gslider->is_entered = 0;
  gslider->timeout_handle = 0;
  gslider->selected = gslider->positions == 1 ? 0 : -1;
  gslider->active = -1;
  gslider->pinned = -1;
  gslider->scale_callback = _default_linear_scale_callback;
  gslider->is_resettable = FALSE;
  gslider->is_entered = FALSE;
  gslider->picker[0] = gslider->picker[1] = gslider->picker[2] = NAN;
  gslider->increment = DTGTK_GRADIENT_SLIDER_DEFAULT_INCREMENT;
  gslider->margin_left = gslider->margin_right =
    GRADIENT_SLIDER_MARGINS_DEFAULT;
  gslider->markers_type = FREE_MARKERS;
  gslider->colors = NULL;
  gslider->min_spacing = 0;
  for(int k = 0; k < gslider->positions; k++)
  {
    gslider->position[k] = 0.0;
    gslider->resetvalue[k] = 0.0;
    gslider->marker[k] = GRADIENT_SLIDER_MARKER_LOWER_FILLED_BIG;
  }
}


// Public functions for multivalue type
GtkWidget *dtgtk_gradient_slider_multivalue_new(const gint positions)
{
  assert(positions <= GRADIENT_SLIDER_MAX_POSITIONS);

  GtkDarktableGradientSlider *gslider;
  gslider = g_object_new(_gradient_slider_get_type(), NULL);
  gslider->positions = positions;
  _gradient_slider_set_defaults(gslider);
  dt_gui_add_class(GTK_WIDGET(gslider), "dt_gslider_multivalue");
  return (GtkWidget *)gslider;
}

GtkWidget *dtgtk_gradient_slider_multivalue_new_with_name
  (const gint positions,
   gchar *name)
{
  GtkWidget *widget = GTK_WIDGET(dtgtk_gradient_slider_multivalue_new(positions));
  if(name) gtk_widget_set_name(widget, name);

  return widget;
}

GtkWidget *dtgtk_gradient_slider_multivalue_new_with_color
  (const GdkRGBA start,
   const GdkRGBA end,
   const gint positions)
{
  assert(positions <= GRADIENT_SLIDER_MAX_POSITIONS);

  GtkDarktableGradientSlider *gslider;
  gslider = g_object_new(_gradient_slider_get_type(), NULL);
  gslider->positions = positions;
  _gradient_slider_set_defaults(gslider);

  // Construct gradient start color
  _gradient_slider_stop_t *gc = g_malloc(sizeof(_gradient_slider_stop_t));
  gc->position = 0.0;
  memcpy(&gc->color, &start, sizeof(GdkRGBA));
  gslider->colors = g_list_append(gslider->colors, gc);

  // Construct gradient stop color
  gc = g_malloc(sizeof(_gradient_slider_stop_t));
  gc->position = 1.0;
  memcpy(&gc->color, &end, sizeof(GdkRGBA));
  gslider->colors = g_list_append(gslider->colors, gc);
  dt_gui_add_class(GTK_WIDGET(gslider), "dt_gslider_multivalue");
  return (GtkWidget *)gslider;
}

GtkWidget *dtgtk_gradient_slider_multivalue_new_with_color_and_name
  (const GdkRGBA start,
   const GdkRGBA end,
   const gint positions,
   gchar *name)
{
  GtkWidget *widget = GTK_WIDGET(dtgtk_gradient_slider_multivalue_new_with_color(start, end, positions));
  if(name) gtk_widget_set_name(widget, name);

  return widget;
}

void dtgtk_gradient_slider_multivalue_set_stop
  (GtkDarktableGradientSlider *gslider,
   const gfloat position,
   const GdkRGBA color)
{
  g_return_if_fail(gslider != NULL);
  const gfloat rawposition = gslider->scale_callback((GtkWidget *)gslider,
                                                     position, GRADIENT_SLIDER_SET);
  // First find color at position, if exists update color, otherwise
  // create a new stop at position.
  GList *current = g_list_find_custom(gslider->colors, (gpointer)&rawposition,
                                      _list_find_by_position);
  if(current != NULL)
  {
    memcpy(&((_gradient_slider_stop_t *)current->data)->color,
           &color, sizeof(GdkRGBA));
  }
  else
  {
    // stop didn't exist lets add it
    _gradient_slider_stop_t *gc = g_malloc(sizeof(_gradient_slider_stop_t));
    gc->position = rawposition;
    memcpy(&gc->color, &color, sizeof(GdkRGBA));
    gslider->colors = g_list_append(gslider->colors, gc);
  }
}

void dtgtk_gradient_slider_multivalue_clear_stops
  (GtkDarktableGradientSlider *gslider)
{
  g_return_if_fail(gslider != NULL);
  g_list_free_full(gslider->colors, g_free);
  gslider->colors = NULL;
}

GType dtgtk_gradient_slider_multivalue_get_type()
{
  return _gradient_slider_get_type();
}

gdouble dtgtk_gradient_slider_multivalue_get_value
  (GtkDarktableGradientSlider *gslider,
   const gint pos)
{
  assert(pos <= gslider->positions);

  return gslider->scale_callback((GtkWidget *)gslider,
                                 gslider->position[pos],
                                 GRADIENT_SLIDER_GET);
}

void dtgtk_gradient_slider_multivalue_get_values
  (GtkDarktableGradientSlider *gslider,
   gdouble *values)
{
  g_return_if_fail(gslider != NULL);
  for(int k = 0; k < gslider->positions; k++)
    values[k] = gslider->scale_callback((GtkWidget *)gslider,
                                        gslider->position[k],
                                        GRADIENT_SLIDER_GET);
}

void dtgtk_gradient_slider_multivalue_set_value
  (GtkDarktableGradientSlider *gslider,
   const gdouble value,
   const gint pos)
{
  g_return_if_fail(gslider != NULL);
  assert(pos <= gslider->positions);

  gslider->position[pos] = CLAMP(gslider->scale_callback((GtkWidget *)gslider,
                                                         value,
                                                         GRADIENT_SLIDER_SET),
                                 0.0, 1.0);
  gslider->selected = gslider->positions == 1 ? 0 : -1;
  if(!DT_IN_GUI_UPDATE()) g_signal_emit_by_name(G_OBJECT(gslider),
                                                  "value-changed");
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

void dtgtk_gradient_slider_multivalue_set_value_pushing
  (GtkDarktableGradientSlider *gslider,
   const gdouble value,
   const gint pos)
{
  g_return_if_fail(gslider != NULL);
  assert(pos <= gslider->positions);

  const gdouble newpos = CLAMP(gslider->scale_callback((GtkWidget *)gslider,
                                                        value,
                                                        GRADIENT_SLIDER_SET),
                               0.0, 1.0);
  // _slider_move (FREE_MARKERS branch) only ever pushes the neighbour that
  // lies on the side newpos is heading towards -- the same direction a mouse
  // drag would be moving in to reach it.
  const gint direction = (newpos < gslider->position[pos]) ? MOVE_LEFT : MOVE_RIGHT;
  _slider_move((GtkWidget *)gslider, pos, newpos, direction);

  gslider->selected = gslider->positions == 1 ? 0 : -1;
  if(!DT_IN_GUI_UPDATE()) g_signal_emit_by_name(G_OBJECT(gslider),
                                                  "value-changed");
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

void dtgtk_gradient_slider_multivalue_set_values
  (GtkDarktableGradientSlider *gslider,
   gdouble *values)
{
  g_return_if_fail(gslider != NULL);
  g_return_if_fail(values != NULL);
  for(int k = 0; k < gslider->positions; k++)
    gslider->position[k] = CLAMP(gslider->scale_callback((GtkWidget *)gslider,
                                                         values[k],
                                                         GRADIENT_SLIDER_SET),
                                 0.0, 1.0);
  gslider->selected = gslider->positions == 1 ? 0 : -1;
  if(!DT_IN_GUI_UPDATE()) g_signal_emit_by_name(G_OBJECT(gslider),
                                                  "value-changed");
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

void dtgtk_gradient_slider_multivalue_set_marker
  (GtkDarktableGradientSlider *gslider,
   const gint mark,
   const gint pos)
{
  g_return_if_fail(gslider != NULL);
  assert(pos <= gslider->positions);

  gslider->marker[pos] = mark;
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

void dtgtk_gradient_slider_multivalue_set_markers
  (GtkDarktableGradientSlider *gslider,
   gint *markers)
{
  g_return_if_fail(gslider != NULL);
  for(int k = 0; k < gslider->positions; k++) gslider->marker[k] = markers[k];
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

void dtgtk_gradient_slider_multivalue_set_resetvalue
  (GtkDarktableGradientSlider *gslider,
   const gdouble value,
   const gint pos)
{
  g_return_if_fail(gslider != NULL);
  assert(pos <= gslider->positions);

  gslider->resetvalue[pos] = gslider->scale_callback((GtkWidget *)gslider,
                                                     value, GRADIENT_SLIDER_SET);
  gslider->is_resettable = TRUE;
}

gdouble dtgtk_gradient_slider_multivalue_get_resetvalue
  (GtkDarktableGradientSlider *gslider,
   const gint pos)
{
  assert(pos <= gslider->positions);

  return gslider->scale_callback((GtkWidget *)gslider,
                                 gslider->resetvalue[pos], GRADIENT_SLIDER_GET);
}

void dtgtk_gradient_slider_multivalue_set_resetvalues
  (GtkDarktableGradientSlider *gslider,
   gdouble *values)
{
  g_return_if_fail(gslider != NULL);
  for(int k = 0; k < gslider->positions; k++)
    gslider->resetvalue[k] = gslider->scale_callback((GtkWidget *)gslider,
                                                     values[k],
                                                     GRADIENT_SLIDER_SET);
  gslider->is_resettable = TRUE;
}

void dtgtk_gradient_slider_multivalue_set_picker
  (GtkDarktableGradientSlider *gslider,
   const gdouble value)
{
  g_return_if_fail(gslider != NULL);
  gslider->picker[0] = gslider->picker[1] = gslider->picker[2]
    = gslider->scale_callback((GtkWidget *)gslider, value, GRADIENT_SLIDER_SET);
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

void dtgtk_gradient_slider_multivalue_set_picker_meanminmax
  (GtkDarktableGradientSlider *gslider,
   const gdouble mean,
   const gdouble min,
   const gdouble max)
{
  g_return_if_fail(gslider != NULL);
  gslider->picker[0] = gslider->scale_callback((GtkWidget *)gslider,
                                               mean, GRADIENT_SLIDER_SET);
  gslider->picker[1] = gslider->scale_callback((GtkWidget *)gslider,
                                               min, GRADIENT_SLIDER_SET);
  gslider->picker[2] = gslider->scale_callback((GtkWidget *)gslider,
                                               max, GRADIENT_SLIDER_SET);
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

gboolean dtgtk_gradient_slider_multivalue_is_dragging
  (GtkDarktableGradientSlider *gslider)
{
  g_return_val_if_fail(gslider != NULL, FALSE);
  return gslider->is_dragging;
}

void dtgtk_gradient_slider_multivalue_set_increment
  (GtkDarktableGradientSlider *gslider,
   const gdouble value)
{
  g_return_if_fail(gslider != NULL);
  gslider->increment = value;
}

void dtgtk_gradient_slider_multivalue_set_scale_callback
  (GtkDarktableGradientSlider *gslider,
   float (*callback)(GtkWidget *self, float value, int dir))
{
  float (*old_callback)(GtkWidget*, float, int) = gslider->scale_callback;
  float (*new_callback)(GtkWidget*, float, int) = (callback == NULL ? _default_linear_scale_callback : callback);
  GtkWidget *self = (GtkWidget *)gslider;

  if(old_callback == new_callback) return;

  for(int k = 0; k < gslider->positions; k++)
  {
    gslider->position[k] = new_callback(self, old_callback(self, gslider->position[k], GRADIENT_SLIDER_GET), GRADIENT_SLIDER_SET);
    gslider->resetvalue[k] = new_callback(self, old_callback(self, gslider->resetvalue[k], GRADIENT_SLIDER_GET), GRADIENT_SLIDER_SET);
  }

  for(int k = 0; k < 3; k++)
  {
    gslider->picker[k] = new_callback(self, old_callback(self, gslider->picker[k], GRADIENT_SLIDER_GET), GRADIENT_SLIDER_SET);
  }

  for(GList *current = gslider->colors; current; current = g_list_next(current))
  {
    _gradient_slider_stop_t *stop = (_gradient_slider_stop_t *)current->data;
    stop->position = new_callback(self, old_callback(self, stop->position, GRADIENT_SLIDER_GET), GRADIENT_SLIDER_SET);
  }

  gslider->scale_callback = new_callback;
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}


// Public functions for single value type
GtkWidget *dtgtk_gradient_slider_new()
{
  GtkWidget *gslider = dtgtk_gradient_slider_multivalue_new(1);
  dt_gui_add_class(gslider, "dt_gslider");
  return gslider;
}

GtkWidget *dtgtk_gradient_slider_new_with_name(gchar *name)
{
  GtkWidget *widget = GTK_WIDGET(dtgtk_gradient_slider_new());
  if(name) gtk_widget_set_name(widget, name);

  return widget;
}

GtkWidget *dtgtk_gradient_slider_new_with_color(const GdkRGBA start,
                                                const GdkRGBA end)
{
  GtkWidget *gslider = dtgtk_gradient_slider_multivalue_new_with_color
    (start, end, 1);
  dt_gui_add_class(gslider, "dt_gslider");
  return gslider;
}

GtkWidget *dtgtk_gradient_slider_new_with_color_and_name(const GdkRGBA start,
                                                         const GdkRGBA end,
                                                         gchar *name)
{
  GtkWidget *widget = GTK_WIDGET(dtgtk_gradient_slider_new_with_color(start, end));
  if(name) gtk_widget_set_name(widget, name);

  return widget;
}

void dtgtk_gradient_slider_set_stop(GtkDarktableGradientSlider *gslider,
                                    const gfloat position,
                                    const GdkRGBA color)
{
  dtgtk_gradient_slider_multivalue_set_stop(gslider, position, color);
}

GType dtgtk_gradient_slider_get_type()
{
  return _gradient_slider_get_type();
}

gdouble dtgtk_gradient_slider_get_value(GtkDarktableGradientSlider *gslider)
{
  return dtgtk_gradient_slider_multivalue_get_value(gslider, 0);
}

void dtgtk_gradient_slider_set_value(GtkDarktableGradientSlider *gslider,
                                     const gdouble value)
{
  dtgtk_gradient_slider_multivalue_set_value(gslider, value, 0);
}

void dtgtk_gradient_slider_set_marker(GtkDarktableGradientSlider *gslider,
                                      const gint mark)
{
  dtgtk_gradient_slider_multivalue_set_marker(gslider, mark, 0);
}

void dtgtk_gradient_slider_set_resetvalue(GtkDarktableGradientSlider *gslider,
                                          const gdouble value)
{
  dtgtk_gradient_slider_multivalue_set_resetvalue(gslider, value, 0);
}

gdouble dtgtk_gradient_slider_get_resetvalue(GtkDarktableGradientSlider *gslider)
{
  return dtgtk_gradient_slider_multivalue_get_resetvalue(gslider, 0);
}

void dtgtk_gradient_slider_set_picker(GtkDarktableGradientSlider *gslider,
                                      const gdouble value)
{
  g_return_if_fail(gslider != NULL);
  gslider->picker[0] = gslider->picker[1] = gslider->picker[2]
    = gslider->scale_callback((GtkWidget *)gslider, value, GRADIENT_SLIDER_SET);
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

void dtgtk_gradient_slider_set_picker_meanminmax
  (GtkDarktableGradientSlider *gslider,
   const gdouble mean,
   const gdouble min,
   const gdouble max)
{
  g_return_if_fail(gslider != NULL);
  gslider->picker[0] = gslider->scale_callback((GtkWidget *)gslider,
                                               mean, GRADIENT_SLIDER_SET);
  gslider->picker[1] = gslider->scale_callback((GtkWidget *)gslider,
                                               min, GRADIENT_SLIDER_SET);
  gslider->picker[2] = gslider->scale_callback((GtkWidget *)gslider,
                                               max, GRADIENT_SLIDER_SET);
  gtk_widget_queue_draw(GTK_WIDGET(gslider));
}

gboolean dtgtk_gradient_slider_is_dragging(GtkDarktableGradientSlider *gslider)
{
  g_return_val_if_fail(gslider != NULL, FALSE);
  return gslider->is_dragging;
}

void dtgtk_gradient_slider_set_increment(GtkDarktableGradientSlider *gslider,
                                         const gdouble value)
{
  g_return_if_fail(gslider != NULL);
  gslider->increment = value;
}


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
