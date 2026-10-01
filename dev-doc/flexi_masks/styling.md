# Styling the masks panel

How to restyle the icons in the masks panel's row and group headers from a
theme or from your CSS tweaks (preferences > general > "modify selected theme
with CSS tweaks below"), without reading the panel's code.

## The rule of thumb

Start every rule with `#masks-list-box`, the panel's list. The theme styles
these parts with classes only, so a rule with that id in front always wins,
whatever state the part is in and wherever your rule sits in the file:

```css
#masks-list-box .mask-group-header .mask-lead.mask-inverted
{
  background-image: none;
  background-color: #e8e8e8;
}
```

That one id is the only thing to remember. The rest is in the tables below.

## Parts

Each part has one class, on the widget that paints it.

| class | part |
|---|---|
| `.mask-lead` | the lead icon, at the left of a header |
| `.mask-drawer` | the box holding the three icons on the right |
| `.mask-expander` | rightmost: shows or hides the row's controls or the group's elements |
| `.mask-eye` | second from the right: click to disable, shift+click to solo |
| `.mask-notes` | third from the right on a group made by a preset: its notes |
| `.mask-picker` | third from the right on a parametric row: its color picker |
| `.mask-link` | third from the right on a linked shape or a raster mask: the chain to the other end |
| `.mask-badge` | left of the drawer: an element or group that is nearly invisible |
| `.mask-channel-eye` | in a parametric row's controls: bypasses one channel range |

## States

Each state is a class or a GTK state on the part itself, so a state rule is
the part's class plus one more.

| part | state | when |
|---|---|---|
| `.mask-lead` | `.mask-inverted` | the element's or group's output is inverted |
| `.mask-lead` | `.mask-channel` | a parametric row: the lead is the channel's code, not a glyph |
| `.mask-eye` | `.mask-soloed` | the element or group is soloed |
| `.mask-eye` | `.mask-disabled` | the element or group is disabled |
| `.mask-expander` | `:checked` | open |
| `.mask-expander` | `:disabled` | an empty group's, with nothing to open |
| `.mask-notes` | `:checked` | the notes are shown |
| `.mask-badge` | `.mask-no-effect` | the element does nothing at all (a channel still covering its whole range), rather than little |
| `.mask-channel-eye` | `:checked` | the range is bypassed |
| any button | `:hover` | under the pointer |

## Element or group

Whose icon it is comes from the header it sits in: `.mask-element-header`
for an element's row (shapes, parametric channels, raster masks and nested
groups alike), `.mask-group-header` for a group's.

```css
/* every inverted lead */
#masks-list-box .mask-lead.mask-inverted { ... }
/* only a group's */
#masks-list-box .mask-group-header .mask-lead.mask-inverted { ... }
/* only an element's */
#masks-list-box .mask-element-header .mask-lead.mask-inverted { ... }
```

## Sizes

Every icon is 18px square, fixed in code. `padding` insets the glyph inside
it and never resizes the icon, so the headers keep their height and the icons
of every row stay lined up. Padding on `.mask-drawer` is the one exception:
it adds room around all three icons, and grows the drawer by as much.

## Backgrounds

Leads and the drawer have no plate: their icons sit on the header. Only an
inverted lead has one, in `mask_handle_inverted_bg`, with an edge drawn as an
inset `box-shadow`. To give every lead a plate:

```css
#masks-list-box .mask-lead
{
  background-color: #3a3a3a;
}
```

A frame around an icon is a `box-shadow` too, such as
`box-shadow: inset 0 0 0 1px red;`. An element's lead is drawn in code and
does not paint a CSS `border`.

## States are yours too

A rule outranks the theme in every state, so it also covers the states you
did not mention. The rule above recolors inverted leads as well, which then
look like the rest. Give each state you want to keep different its own rule;
having one more class, it wins over your plain one:

```css
#masks-list-box .mask-lead
{
  background-color: #3a3a3a;
}

#masks-list-box .mask-lead.mask-inverted
{
  background-color: #e8e8e8;
  color: #202020;
}
```

## Colors alone

To change colors only, and keep every state's look, redefine the theme's color
tokens instead. Your CSS tweaks are loaded after the theme, into the same
stylesheet, so a redefinition replaces the token everywhere it is used.

| token | used for |
|---|---|
| `mask_rest` | the headers at rest |
| `mask_implied` | the header of a group holding the selection |
| `mask_selected_top`, `mask_selected` | a selected header, top and bottom of its gradient |
| `mask_hover_top`, `mask_hover` | a hovered header, top and bottom of its gradient |
| `mask_card` | the ground of a row's open controls |
| `mask_list_bg` | the ground of the whole list |
| `mask_handle_fg` | the glyph of a lead icon |
| `mask_handle_inverted_bg`, `mask_handle_inverted_fg` | an inverted lead's plate, and its glyph and edge |
| `mask_text_rest`, `mask_text_implied`, `mask_text_selected`, `mask_text_hover` | header names, per header state |

```css
@define-color mask_handle_inverted_bg #f0d070;
@define-color mask_handle_inverted_fg #202020;
```

The headers themselves (their shades and rails) have no classes of their own
to target: restyle them through these tokens. A rail takes its header's
shade, the bottom one of a gradient.
