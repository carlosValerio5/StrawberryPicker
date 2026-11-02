# Labeling rubric

How a person decides each label in [data-contract.md](../data-contract.md). Everyone uses
it: test-set labelers, reviewers of `needs_review` rows, and anyone auditing a public
source. Two people with this page and the same photo should give the same answer.
Where they don't, the agreement score tells us so (ADR [0005](../decisions/0005-acceptance-criteria.md)).

**External reference:** USDA AMS, *United States Standards for Grades of Strawberries*
(effective 2006-02-23, §51.3115–51.3124). We take the color threshold and the defect
definitions from it, so the labels match a published standard rather than our own
taste (ADR [0004](../decisions/0004-usda-label-rubric.md)).

## Label one berry at a time

1. Look only at the berry inside the box, or the centered berry.
2. Judge the **surface you can see**. A photo shows about half the berry. Assume the
   hidden side looks like the visible one, unless the shoulders near the cap show
   otherwise.
3. Decide maturity, then defects.
4. If you can't decide (blur, glare, too small, color cast from the light), set
   **`unsure`**. Don't guess. An `unsure` row goes to adjudication, not into gold.

## Maturity (pick one)

| Label | Rule | USDA anchor |
|---|---|---|
| `unripe` | **Less than ¾ of the visible surface is pink or red.** Green, white and "turning" berries count, and so do berries red at the tip but white at the shoulders | U.S. No. 1 requires "not less than three-fourths of its surface showing a pink or red color" (§51.3115) |
| `ripe` | **At least ¾ of the surface is pink or red**, and none of the overripe signs below apply. Pink counts. You don't need deep red | §51.3115 |
| `overripe` | Pink or red, plus **two or more** of: dark red to purple-maroon color, dull or matte skin that has lost its gloss, wrinkled or sunken skin, a flattened or slumped shape, leaking juice, seeds standing proud because the flesh has shrunk | "Overripe" means dead ripe, becoming soft, … unfit for shipment (§51.3120) |

How to estimate "¾ of the surface": split the visible berry into four vertical strips,
tip to cap. It's `ripe` only if at least three strips are mostly pink or red.

Softness can't be seen. The USDA definition is about softness, so the photo rule above
is a **visual proxy**. Its error is one of the things the test-set agreement measures.

## Defects (pick zero or more)

Mark a defect if you can see it at all, however small. Severity isn't labeled in v0
(see "Known gaps").

| Label | Visible sign | USDA anchor |
|---|---|---|
| `mold` | Fuzzy growth: grey-brown (*Botrytis*), white powdery, or black spots with fuzz | U.S. No. 1 must be "free from mold" (§51.3115) |
| `rot` | Decay without fuzz: brown water-soaked patches, collapsed or leaking flesh | "Decayed or leaky berries" are serious damage (§51.3123 d) |
| `bruise` | Darker, glassy or water-soaked area, often flattened, with skin unbroken | "Badly bruised berries" (§51.3123 c) |
| `deformed` | Clearly misshapen: cat-facing, a hard seedy tip, or a "button" berry | "Badly deformed berries" (§51.3123 b). "Undeveloped" / "button" (§51.3121) |
| `pest_damage` | Holes, bite marks, slug trails, insect frass | "damage caused by … insects" (§51.3115) |
| `mechanical_damage` | Cuts, crushed areas, torn skin, a punctured surface | "damage caused by … mechanical or other means" (§51.3115) |

No defects means `defects: []`. Use that only after looking. If the photo can't show
defects (for example the back is hidden and the front is clean), record what you can
see, and set `unsure` only if a defect is half-visible.

## Our grade compared with the USDA grade

`grade_from_defects` (reject when mold or rot is present, standard for any other defect,
premium when there are none) is **defect-only**. The USDA grade also uses color and
severity:

| Ours | Roughly matches | Difference |
|---|---|---|
| `premium` | U.S. No. 1 | USDA also needs ≥ ¾ pink or red, the cap attached, and diameter ≥ ¾ in |
| `standard` | U.S. No. 2 | USDA No. 2 still bans *serious* damage: soft, badly bruised or badly deformed berries |
| `reject` | below U.S. No. 2 | USDA also rejects berries with less than ½ of the surface colored |

## Known gaps (not labeled in v0)

These are recorded so nobody assumes the model handles them. Changing any of them is a
taxonomy change and needs an ADR:
- **Severity:** USDA separates "damage" from "serious damage".
- **Dirt** and foreign matter.
- **Missing cap:** U.S. No. 1 requires the cap to be attached.
- **Size:** needs a scale reference in the photo.

## Reference images

To be added from the audit contact sheets (`reports/dataset-audit/<source>/`): three
clear examples per maturity class and one per defect, each with its source and
`bronze_path`. Only images whose license allows redistribution may be committed (as
small thumbnails in `docs/labeling/img/`). Otherwise, link to the bronze path.
