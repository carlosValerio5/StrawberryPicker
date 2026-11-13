---
name: StrawberryPicker
description: A phone photo of a tray, filed as a one-tray market report on green-bar fanfold.
colors:
  desk: "#252a26"
  paper: "#f7f9f5"
  paper-hi: "#fcfdfa"
  band: "#ebf3e8"
  band-row: "#dcebd8"
  band-hot: "#c8ddc2"
  rule: "#a7bca3"
  rule-strong: "#7f977b"
  ink: "#132017"
  ink-lift: "#26402e"
  ink-2: "#384a3d"
  ink-3: "#536456"
  ripe: "#b3102b"
  unripe: "#56730a"
  overripe: "#5a1939"
  stamp: "#c93717"
  stamp-deep: "#8a2410"
  stamp-wash: "#fbe9e2"
  photo-reject: "#ff6a45"
  premium: "#1d5631"
typography:
  display:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "62px"
    fontWeight: 820
    lineHeight: "64px"
    letterSpacing: "0.002em"
    fontVariation: "'wdth' 66"
  headline:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "34px"
    fontWeight: 800
    lineHeight: "40px"
    letterSpacing: "normal"
    fontVariation: "'wdth' 70"
  title:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "28px"
    fontWeight: 780
    lineHeight: "32px"
    letterSpacing: "0.01em"
    fontVariation: "'wdth' 74"
  stamp:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "27px"
    fontWeight: 860
    lineHeight: "34px"
    letterSpacing: "0.1em"
    fontVariation: "'wdth' 92"
  wordmark:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "19px"
    fontWeight: 850
    lineHeight: "24px"
    letterSpacing: "0.015em"
    fontVariation: "'wdth' 118"
  body:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "17px"
    fontWeight: 460
    lineHeight: "24px"
    letterSpacing: "normal"
  body-sm:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: "20px"
    letterSpacing: "normal"
  button:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 750
    lineHeight: "24px"
    letterSpacing: "0.05em"
    fontVariation: "'wdth' 92"
  data:
    fontFamily: "Martian Mono, ui-monospace, Menlo, monospace"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: "24px"
    letterSpacing: "0.02em"
    fontFeature: "'tnum' 1"
    fontVariation: "'wdth' 87.5"
  label:
    fontFamily: "Martian Mono, ui-monospace, Menlo, monospace"
    fontSize: "12.5px"
    fontWeight: 700
    lineHeight: "24px"
    letterSpacing: "0.05em"
    fontVariation: "'wdth' 87.5"
  label-sm:
    fontFamily: "Martian Mono, ui-monospace, Menlo, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: "24px"
    letterSpacing: "0.04em"
    fontVariation: "'wdth' 87.5"
rounded:
  tag: "2px"
  control: "3px"
  field: "4px"
  stamp: "6px"
  dot: "50%"
spacing:
  hair: "4px"
  half: "12px"
  line: "24px"
  line-and-half: "36px"
  band: "48px"
  feed: "40px"
  feed-narrow: "18px"
components:
  press:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper-hi}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    padding: "0 22px 0 18px"
    height: "48px"
  press-hover:
    backgroundColor: "{colors.ink-lift}"
  press-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.control}"
    height: "48px"
  press-quiet-hover:
    backgroundColor: "{colors.band-hot}"
  fileline-action:
    backgroundColor: "{colors.paper-hi}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "0 14px 0 11px"
    height: "36px"
  tray:
    backgroundColor: "{colors.paper-hi}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "0 8px"
    height: "30px"
    width: "40px"
  tray-current:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper-hi}"
  lot-tag-ripe:
    backgroundColor: "{colors.ripe}"
    textColor: "#ffffff"
    rounded: "{rounded.tag}"
    padding: "0 7px"
    height: "22px"
  lot-tag-unripe:
    backgroundColor: "{colors.unripe}"
    textColor: "#ffffff"
  lot-tag-overripe:
    backgroundColor: "{colors.overripe}"
    textColor: "#ffffff"
  stamp-reject:
    textColor: "{colors.stamp}"
    typography: "{typography.stamp}"
    rounded: "{rounded.stamp}"
    padding: "3px 12px 1px"
  stamp-premium:
    textColor: "{colors.premium}"
  stamp-standard:
    textColor: "{colors.ink-2}"
  lot-row:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.data}"
    height: "48px"
  lot-row-odd:
    backgroundColor: "{colors.band-row}"
  lot-row-on:
    backgroundColor: "{colors.band-hot}"
  berry-slip:
    backgroundColor: "{colors.paper-hi}"
    textColor: "{colors.ink}"
    width: "16rem"
  errata:
    backgroundColor: "{colors.stamp-wash}"
    textColor: "{colors.stamp-deep}"
    padding: "10px 14px"
---

# Design System: StrawberryPicker

## Overview

**Creative North Star: "The Market News Wire"**

Every result is filed as a printed commodity report on green-bar continuous fanfold, the way a market news terminal reports a lot: masthead and dateline, a monumental verdict headline, a condition line, a printed tally, then one lot line per berry. The sheet is cool paper white banded in pale green two text lines deep, with a punched tractor-feed strip down each edge and a dashed, tick-punched perforation where the photo plate folds against the report. The page sits on a dark desk that shows through the punched holes.

Ink is dense green-black and contrast is high, because the room is a half-lit meeting room with a projector washing out blacks. Headlines are a narrow wire-service grotesk set in heavy uppercase; everything tabular, coded or labelled is a narrowed mono with tabular figures. Ripeness is carried twice, by ink and by line form, so it survives a washed-out projector and colour-blind viewers. A reject is a vermilion rubber stamp, askew and grained. Rank comes only from layering: the photo is a print laid on the sheet, and a berry's detail is a torn slip laid over the photo.

Nothing moves except paper and ink: the report feeds in line by line when a result arrives, and the stamp lands once. Reduced motion removes both.

**Key Characteristics:**
- Green-bar fanfold ground: 48px bands alternating with paper, on a 24px text line.
- Tractor-feed strips with desk-coloured punched holes on the 24px pitch.
- Condensed heavy uppercase Archivo for verdicts; narrowed Martian Mono for every code, label and figure.
- Three ripeness inks, each paired with a line form: solid, dashed, double.
- Grades as ink: vermilion stamp for reject, deep green for premium, ink-2 for standard.
- Depth by layering paper on paper; no cards, no tiles.

## Colors

A cool green-paper ground with dense green-black ink, three ripeness inks and two grade inks; colour always carries a meaning, never decoration.

### Primary
- **Market Ink** (ink): all body text, rules under the masthead and table head, filled press buttons, the current tray, lot tags with no ripeness. Hover on a filled press lifts to **Lifted Ink** (ink-lift).
- **Pack-out Green** (premium): the premium grade (text, stamp, grade bar), the ready state of the wire dot, the focus ring and native `accent-color`.

### Secondary
- **Strawberry Red** (ripe): ripe berries. Solid line form.
- **Chartreuse Leaf** (unripe): unripe berries. Dashed line form.
- **Aubergine** (overripe): overripe berries. Double line form.

### Tertiary
- **Vermilion Stamp** (stamp): reject only: the rubber stamp, mini-stamps in the lot table and legend, reject defects text, the hatched reject segment of the grade bar, error and offline wire dot, the errata border.
- **Stamp Wash** (stamp-wash) and **Deep Vermilion** (stamp-deep): the errata slip's ground and text.
- **Photo Reject** (photo-reject): the reject outline drawn on the photo, deliberately brighter than the paper vermilion so it reads on red fruit over a dark halo.

### Neutral
- **Desk** (desk): the surface the sheet lies on; shows through tractor holes, behind crop and sample photos, the drop veil.
- **Fanfold Paper** (paper) and **Bright Leaf** (paper-hi): the sheet, and the brighter stock of anything laid on it (photo mount, slip, samples, buttons, fields).
- **Green Bar** (band): the 48px bands of the fanfold. **Row Bar** (band-row): odd lot rows. **Hot Bar** (band-hot): hover and active rows, button hover, selection.
- **Rule** (rule) and **Strong Rule** (rule-strong): hairlines inside slips and the legend; perforations, dashed borders, idle tray borders, scrollbar.
- **Ink 2** (ink-2) and **Ink 3** (ink-3): secondary text (dateline, captions, labels), and the faintest text (table head, slip fact labels, tally leaders).

### Named Rules
**The Two Carriers Rule.** Ripeness is never colour alone. Every place a ripeness ink appears as a mark (photo outline, legend swatch, lot-row swatch, mix bar segment) it carries its line form too: ripe solid, unripe dashed, overripe double.

**The Stamp Is Only Reject Rule.** Vermilion means reject and nothing else. It is not an accent, a link colour or a highlight.

**The Bands Are the Grid Rule.** The green bands are the page's measuring grid, 48px deep on a 24px line. Do not tint, recolour or shift them per section.

## Typography

**Display Font:** Archivo, variable width 62–125% and weight 100–900, self-hosted (with Helvetica Neue, Arial)
**Body Font:** Archivo at normal width
**Label/Mono Font:** Martian Mono, variable width 75–112.5%, self-hosted (with ui-monospace, Menlo)

**Character:** a wire-service grotesk squeezed narrow and set heavy for verdicts, against a slightly narrowed dot-matrix-flavoured mono for everything that a terminal would print. Width carries the voice: verdicts and ripeness names are condensed (66–74%), the stamp and buttons are near-normal (92%), and only the wordmark is expanded (118%).

### Hierarchy
- **Display** (820, 62px/64px, width 66%, uppercase, balanced wrap): the verdict headline ("Reject this tray."), the empty-form headline, the printing headline in ink-2. 44px/48px below 960px.
- **Headline** (800, 34px/40px, width 70%, uppercase): the ripeness name on the berry slip, in its ripeness ink. The drop veil uses the same voice at 44px/48px.
- **Title** (780, 28px/32px, width 74%, uppercase): the empty-plate intake title.
- **Stamp** (860, 27px/34px, width 92%, 0.1em tracking, uppercase): the rubber stamp. 16px/24px below 960px.
- **Wordmark** (850, 19px, width 118%, uppercase): the masthead name only.
- **Body** (460, 17px/24px, max 62ch): the condition line and its explanations.
- **Body small** (600, 15px): drop strip, sample titles, legend definitions (15px/20px), errata (560, 15px/22px).
- **Button** (750, 15px, width 92%, 0.05em, uppercase): press buttons; 14px in the file line.
- **Data** (13px, width 87.5%, 0.02em, uppercase, tabular figures): lot lines; codes at 700, ripeness and grade at 600. Table head at 11.5px/600/0.06em in ink-3.
- **Label** (12.5px, 600–700, width 87.5%, 0.04–0.06em, uppercase): dateline, wire, file line, rubrics, tally, mix counts, lot tags, condition lead word.
- **Label small** (12px, width 87.5%, uppercase): captions, slip codes and fact labels, mix labels, notes, elapsed time.

### Named Rules
**The Terminal Prints It Rule.** Codes, counts, percentages, captions and labels are Martian Mono, uppercase, at width 87.5%, with tabular figures where numbers align. Prose and verdicts are never mono.

**The Width Is the Voice Rule.** Hierarchy is made with font-stretch and weight together, not size alone: verdict condensed and heavy, stamp near-normal and spaced, wordmark expanded. Do not set display text at normal width.

## Layout

The page is one sheet of fanfold filling the viewport (100dvh), with the desk visible only through the punched holes. On desktop it is a four-column grid: a 40px tractor strip, the photo plate (1.1fr), the report (at least 30rem, 1fr), and a second 40px strip. The plate and report fold against each other along a dashed perforation punched with ticks on the same 24px pitch as the feed holes. The plate is fixed and full height; the report scrolls inside the sheet with its bands attached to the content, so the paper scrolls rather than the window.

Rhythm is a 24px line and a 48px band. Text blocks are spaced in whole lines (24px, 36px, 48px); table rows, press buttons, legend rows, the plate caption and the drop strip are one band (48px) tall; tally rows are one line each. The report's left gutter is a line and a half (36px), its right gutter one line.

Below 960px the report comes first and the plate follows: the sheet becomes a single column between 18px tractor strips with smaller (4px) holes, the window scrolls instead of the report, the fold becomes a rule above the plate, the tally collapses to one column, mix counts drop under their bars, the masthead stacks, and the berry slip opens in flow as a full-width row under its lot line.

### Named Rules
**The Whole Lines Rule.** Vertical space is spent in multiples of the 24px line; anything that can be one band tall is exactly 48px.

## Elevation & Depth

Depth is paper laid on paper. The sheet itself is flat; three kinds of leaf sit above it, each with a tight contact shadow plus a long, pulled-in drop, and a defined edge because they are cut stock: the photo print, the sample prints on the contact sheet, and the berry slip (the highest, also rotated -0.6deg with a perforated top edge). Rubber stamps use multiply blending and an SVG turbulence grain instead of shadow. Lot tags carry a 1px under-edge. There is no other elevation.

### Shadow Vocabulary
- **Print** (`box-shadow: 0 1px 2px rgb(19 32 23 / 0.14), 0 14px 32px -14px rgb(19 32 23 / 0.5)`): the photo mounted on the plate.
- **Sample print** (`box-shadow: 0 1px 2px rgb(19 32 23 / 0.12), 0 10px 22px -14px rgb(19 32 23 / 0.45)`), lifting on hover or focus to (`0 2px 4px rgb(19 32 23 / 0.14), 0 18px 30px -14px rgb(19 32 23 / 0.5)`) with a 3px rise.
- **Slip** (`box-shadow: 0 2px 4px rgb(19 32 23 / 0.12), 0 22px 44px -16px rgb(19 32 23 / 0.55)`): the torn berry slip. In flow on narrow screens it keeps only the contact shadow.
- **Tag edge** (`box-shadow: 0 1px 0 rgb(0 0 0 / 0.25)`): lot tags on the photo.
- **Double rule** (`box-shadow: 0 3px 0 -2px #f7f9f5, 0 4px 0 -2px #132017`): the masthead's second underline.

### Named Rules
**The Layering Is Rank Rule.** The only way to raise something is to lay a leaf of brighter stock on top of it. Panels, cards and tinted containers are not used to rank content.

## Shapes

Square paper, barely softened. Controls are 3px; lot tags 2px; the dashed intake frame 4px; stamps 6px with a doubled inner rule. Round forms are reserved for things that are round in the world: punched holes, perforation ticks, the wire status dot and the slip's close button. Borders are 1–1.5px solid ink for fields and prints, dashed rule-strong for anything that is a perforation or a place to drop. Bars are dotted or hatched rather than filled where the meaning needs a second carrier: confidence bars are a dot-matrix row, tally leaders are dotted, the unripe mix segment is dashed, the overripe segment is a double stripe, the reject grade segment is hatched.

## Components

### Masthead
Wordmark at left, the model wire at right (a 9px dot: strong rule idle, amber blinking while warming, pack-out green when ready, vermilion on error or offline), the dateline across the full width under them in ink-2, closed by a double ink rule.

### File line
A printed form field: a mono "FILE" label, the file name on a 1.5px ink underline (14px mono, 600, ellipsized), and a 36px outlined "New photo" action with an upload glyph in SVG. Hover fills the action with hot bar.

### Tray picker
A mono "TRAYS" label followed by 30px-tall numbered keys, 1px strong-rule border on bright leaf. Hover inks the border and fills hot bar; the current tray is solid ink with bright-leaf numerals.

### Verdict headline and rubber stamp
The display headline states the action ("Ship as premium.", "Sort before shipping.", "Pull 2 berries before shipping.", "Reject this tray."). The stamp floats right of it: uppercase stamp type inside a 3px border with a doubled inner rule, rotated -7deg, multiplied, grained by the `ink-grain` turbulence filter. Its ink is the grade: vermilion reject, green premium, ink-2 standard.

### Condition line
Body text (17px, 62ch max) led by a bold mono uppercase word ("CONDITION").

### Leader tally
Two columns of three 24px rows, flowing by column: an ink-2 mono label, a dotted ink-3 leader, a right-aligned bold tabular figure. The ripeness column puts a 22px line-form swatch before each label. One column below 960px.

### Labelled mix bars
A three-column grid: mono label, an 8px ink-bordered bar on bright leaf, a bold mono count. Segments are separated by 1px bright-leaf gaps and each carries its line form: ripe solid, unripe dashed, overripe double stripe; premium solid green, standard solid ink-2, reject hatched vermilion with a vermilion inset edge.

### Lot table
A mono caption rubric with a right-floated note, an 11.5px ink-3 head on an ink rule, then 48px rows alternating paper and row bar. Each row: a line-form swatch and bold lot code, the ripeness in its ink (unripe text deepens to #445c06 so 13px stays at 4.5:1 on the bands), defects in ink-2 (stamp-deep bold when they cause a reject, for the same reason), the grade right-aligned (premium green, reject as a mini-stamp: 2px vermilion border, -4deg, grained). Hover or the linked berry fills hot bar; the active row also gets inset ink rules top and bottom.

### Photo outlines and lot tags
Every berry outline is a 6px dark halo under a 2.5px white line, so it reads on any fruit. Unripe outlines are dashed (9/6); overripe outlines add a second 1.5px inner rectangle inset 5px; rejected outlines turn photo-reject. The linked outline thickens (4px line, 9px halo) and the others drop to 30% opacity. Each outline carries a 22px lot tag button in its ripeness ink, mono 12.5px/600 white, with a white "REJECT" chip in vermilion for rejects; tags lift 2px on hover and flip below the box near the top edge. A "Model view" checkbox (M) hides outlines and tags to show the raw photo.

### Berry slip
A 16rem torn leaf of bright stock laid next to the berry on the photo, rotated -0.6deg with a perforated top edge: a close-up crop in an ink frame on desk, then the lot code (label small), the ripeness name (headline, ripeness ink), a fact list (Defects, Grade) on rule hairlines, and a collapsed "Model confidence" disclosure with dot-matrix bars and right-aligned percentages. It tries right, left, below, then above the berry.
- **Compact variant:** when no full slip fits, the crop is dropped and the close button becomes a transparent ink cross.
- **Mobile in-flow row:** below 960px the slip opens as a full-width row under its lot line, unrotated, unanimated, with only a contact shadow, set back in sans at normal case.

### Contact sheet
Before any report, the plate shows sample trays as prints: a mono head on an ink rule, the drop strip, then an auto-filling grid (min 13.5rem) of bright-leaf prints with a 4:3 photo, a mono number, a title in ink-2 and a mono meta line (reject in vermilion, premium in green). Every first and second of three prints is rotated -0.5deg and 0.4deg; hover or focus straightens and lifts them.

### Drop strip and intake field
The drop strip is a 48px-tall dashed strong-rule field on translucent bright leaf with an upload glyph and a right-aligned mono note. The empty plate's intake field is the same idea at full size: a 2px dashed frame, a 40px upload glyph, the intake title and a mono sub-line. Hover or a drag over the sheet inks the border and brightens the field; a drag also lays a desk-coloured veil over the sheet with a dashed "Release to file this photo".

### Tractor-feed strips and perforated fold
Paper strips 40px wide (18px narrow) punched with desk-coloured 12px holes (8px narrow) on the 24px pitch, separated from the sheet by dashed strong-rule tear lines. The fold between plate and report is a dashed rule overlaid with punched ticks on the same pitch.

### Press buttons
48px tall, 3px corners, 1.5px ink border, button type, an SVG glyph and an optional key hint. Filled: ink, lifting to lifted ink on hover. Quiet: transparent, hot bar on hover. Active presses down 1px; disabled is 45% opacity.

### Printing state
The quiet display headline with a blinking block caret, the condition line explaining the wait, and six ghost lot lines of faint dots with ragged lengths, swept by a print-head band that steps down the sheet.

### Errata slip
Errors print as a slip: stamp-wash ground, 1.5px vermilion border, deep vermilion 15px text led by a bold mono word, rotated -0.4deg, max 60ch.

### Motion
- **Paper feed:** when a report files, the verdict row, condition, tally, mixes, caption, head and each lot row reveal top-down (clip from the top, 6px drop) over 340ms on the ease-out curve, staggered 55ms per line; photo outlines and tags fade in on the same stagger, 120ms behind.
- **Stamp impact:** the stamp lands once, 140ms after the feed starts: from 1.4x and -3deg, overshooting to 0.96x and -8deg, settling at 1x and -7deg over 280ms.
- **Small state motion:** 140–180ms ease-out on hovers and slip entry (220ms drop of 8px); stepped blinks for the caret and warming dot; the print head steps six bands in 2.4s.
- **Reduced motion:** every animation and transition is removed and the print replay is skipped.

## Do's and Don'ts

### Do:
- **Do** put every block on the 24px line and make row-level elements exactly one 48px band tall.
- **Do** pair every ripeness ink with its line form: ripe solid, unripe dashed, overripe double.
- **Do** set codes, counts, labels and captions in Martian Mono, uppercase, width 87.5%, tabular figures for aligned numbers.
- **Do** state the verdict as an action in the condensed display headline and let the stamp carry the grade ink.
- **Do** raise detail by laying a leaf of bright stock (paper-hi) over the sheet, with the slip shadow and a slight rotation.
- **Do** draw photo outlines with a dark halo under a light line so they read on any fruit.
- **Do** keep motion to paper feed and stamp impact, and remove all of it under reduced motion.
- **Do** put the report before the plate below 960px, and open the berry slip in flow under its lot line.

### Don't:
- **Don't** use vermilion for anything other than reject.
- **Don't** show ripeness by colour alone, in any mark, bar or legend.
- **Don't** rank content with cards, stat tiles, tinted panels or a top app bar; the sheet and its leaves are the only surfaces.
- **Don't** set verdicts or ripeness names at normal width, or set prose in mono.
- **Don't** show model confidence up front; it lives in the slip's closed disclosure.
- **Don't** add motion beyond the paper feed, stamp impact and the small state transitions listed in Components.
