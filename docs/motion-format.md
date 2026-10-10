# The Wicroduck motion file

A motion file describes what the Microduck's fourteen joints should be doing
over time. It is plain JSON, it is the only input the motion tools take, and it
is designed so that an AI agent handed *this document* and a request — "make
the duck take a slow bow" — can write a valid one without seeing the code.

If you are that agent: read to the end of **Writing a file**, then write JSON.
Everything after that is context.

---

## The shape

```jsonc
{
  "format": 1,
  "name": "bow",
  "description": "Dip the head forward, hold, then rise.",
  "loop": false,
  "units": "rad",          // or "deg"
  "interp": "cubic",       // or "linear"
  "joints": ["neck_pitch", "head_pitch"],
  "keyframes": [
    { "t": 0.0, "pose": [0.3491,  0.3491] },
    { "t": 1.0, "pose": [0.9500, -0.0500] },
    { "t": 2.0, "pose": [0.9500, -0.0500] },
    { "t": 3.0, "pose": [0.3491,  0.3491] }
  ]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `format` | no | Always `1`. |
| `name` | no | Shown in the UI and used as the filename. |
| `description` | no | One line, for humans. |
| `loop` | no | `false` (default) plays once and holds the last pose; `true` cycles. |
| `units` | no | `"rad"` (default) or `"deg"`. Applies to every angle in the file. |
| `interp` | no | `"cubic"` (default) or `"linear"`. |
| `joints` | no | Which joints this file drives. Omit to mean all fourteen, in the order below. |
| `keyframes` | **yes** | At least one. `t` is seconds; `pose` has one angle per entry in `joints`. |

**Only the joints you list are yours.** Everything else holds its reference
angle for the whole motion. A nod is a one-joint file; you never have to write
fourteen numbers to move one thing.

A single keyframe is a static pose. That is a legitimate motion file.

## The joints

Policy order. `default` is the reference standing pose — the angle a joint
holds when nothing asks otherwise, and the sensible value to start and end a
motion on.

| # | Name | Range (rad) | Range (deg) | Default (rad) |
| --: | --- | --- | --- | --: |
| 0 | `left_hip_yaw` | −0.436 … 0.524 | −25 … 30 | 0 |
| 1 | `left_hip_roll` | −0.384 … 0.384 | −22 … 22 | −0.0873 |
| 2 | `left_hip_pitch` | −1.571 … 1.571 | −90 … 90 | −0.4579 |
| 3 | `left_knee` | −1.571 … 1.571 | −90 … 90 | −0.0049 |
| 4 | `left_ankle` | −1.571 … 1.571 | −90 … 90 | 0.4530 |
| 5 | `neck_pitch` | −1.571 … 1.047 | −90 … 60 | 0.3491 |
| 6 | `head_pitch` | −1.571 … 1.571 | −90 … 90 | 0.3491 |
| 7 | `head_yaw` | −2.967 … 2.967 | −170 … 170 | 0 |
| 8 | `head_roll` | −0.436 … 0.436 | −25 … 25 | 0 |
| 9 | `right_hip_yaw` | −0.524 … 0.436 | −30 … 25 | 0 |
| 10 | `right_hip_roll` | −0.384 … 0.384 | −22 … 22 | 0.0873 |
| 11 | `right_hip_pitch` | −1.571 … 1.571 | −90 … 90 | 0.4579 |
| 12 | `right_knee` | −1.571 … 1.571 | −90 … 90 | 0.0049 |
| 13 | `right_ankle` | −1.571 … 1.571 | −90 … 90 | −0.4530 |

### Which way is which

Measured on the model, not guessed. The duck faces **−X**.

- `neck_pitch` **+** leans the head forward and down — this is the bow joint.
  **−** lifts it back and up.
- `head_pitch` **−** tips the beak down, as if looking at its own feet.
  **+** looks up.
- `head_yaw` **+** turns to the duck's right, **−** to its left.
- `head_roll` tilts the head sideways. Small range; good for a quizzical look.
- **The two legs are mirrored.** Compare `left_hip_pitch` (−0.458) with
  `right_hip_pitch` (+0.458): the same physical motion needs opposite signs.
  To fold a leg — the knee bend of a squat — the left knee goes **positive**
  and the right knee goes **negative**.
- A fully seated duck is at roughly `hip_roll 0, hip_pitch ∓0.408, knee ±1.35,
  ankle 0` (upper sign left, lower sign right). Anything between that and the
  default is a crouch of some depth.

## Writing a file

1. **Start and end at the default pose** unless you mean otherwise. For a
   looping motion the last keyframe must repeat the first exactly, or the file
   is rejected — the loop period is the last `t`, so a mismatch would be a jump
   with no defined duration.
2. **The first keyframe must be at `t: 0`.**
3. **Keep angles inside the range.** Out of range is an error, not a clamp.
4. **Use `"units": "deg"` if you are thinking in degrees.** Writing `45` while
   the file says radians is the single most common way to produce a file that
   is both valid-looking and wrong, so the parser checks for it.
5. **Give it time.** Roughly 0.3 s is quick, 1 s is deliberate, 3 s is slow.
   A motion asking for a 1.5 rad swing in 0.2 s is asking for a servo the duck
   does not have.
6. **Mirror the legs** when you want a symmetric motion. See above.

## What happens to the file

- **Preview** plays it kinematically in the Motion maker — the joints follow
  the file exactly, no physics. This is the view for checking that a motion
  looks the way you meant. Paste your JSON into the panel at the bottom of that
  workspace to load it without saving a file first.
- **Physics playback** runs the same angles through the actuators against
  gravity. Expect it to fall over. That is not a bug in your file: with the
  controls driven open-loop this robot topples in about a second no matter what
  you ask of it, because standing is an active behaviour, not a pose.
- **Training** turns the file into a reward. A policy is scored on matching
  your joint angles at each moment *while keeping its balance*, and it is that
  policy — not the file — that ends up performing the motion on a real duck.

Which leads to the one thing worth knowing before you write anything
ambitious:

> **An authored motion can be dynamically impossible.** The duck's head is
> about 38% of its body mass. A pose that looks right in an editor may not be
> reachable while balancing, and the policy will fail in ways that look like a
> training bug. Prefer motions that keep the mass over the feet, and treat a
> deep squat or a big lean as a genuine athletic request.

## Errors you might see

The parser refuses a file rather than guessing, and every message names the
fix. The ones worth anticipating:

| Message | Cause |
| --- | --- |
| `the first keyframe must be at t = 0` | The timeline has to start at zero. |
| `loop is true but the last keyframe does not repeat the first pose` | Close the cycle, or set `loop: false`. |
| `pose has N angles but M joints are declared` | `pose` length must equal the `joints` length in every keyframe. |
| `drives "X" to V units, outside its range` | Check the table above — and check `units`. |
| `unknown joint "X"` | Only the fourteen names above exist. |
| `unknown field "X"` | A typo in a field name; the file is not silently ignoring it. |

## A complete worked example

A looping squat, all eight leg joints, with the mirrored signs done properly:

```json
{
  "format": 1,
  "name": "squat",
  "description": "Fold halfway to a sit and rise again, keeping the trunk level.",
  "loop": true,
  "units": "rad",
  "interp": "cubic",
  "joints": [
    "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"
  ],
  "keyframes": [
    { "t": 0.0, "pose": [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    { "t": 0.9, "pose": [-0.0436, -0.4329,  0.6725, 0.2265, 0.0436, 0.4329, -0.6725, -0.2265] },
    { "t": 1.8, "pose": [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] }
  ]
}
```

Note the shape of it: keyframe 0 and keyframe 2 are the default pose, verbatim
and identical; keyframe 1 is partway toward the seated angles, with the left
knee positive and the right knee negative.

---

Built-in motions live in `src/motion/library.ts` and are worth reading as
further examples; the Motion maker can open any of them as a starting point.
`node scripts/check-motion.ts` validates the format, the interpolator and the
reward stack together.
