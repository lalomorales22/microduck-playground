// Motions that ship with the app.
//
// These are written as literal FILES rather than built from DEFAULT_POSE in
// code, because their second job is to be the worked examples an author — or
// an agent reading docs/motion-format.md — copies. A file that leans on a
// constant nobody outside this repo has is a bad example.
//
// The duck faces -X. These sign conventions were measured on the compiled
// model:
//
//   neck_pitch  +  leans the head forward and down (a bow); - lifts it back
//   head_pitch  -  tips the beak down (looking at its feet); + looks up
//   head_yaw    +  turns to its right
//   knee        +  on the left, - on the right folds the leg (the two legs'
//                  axes are mirrored — see DEFAULT_POSE's sign pattern)
//
// The leg angles in "squat" and "bow" are fractions of the way to the seated
// pose in standup.ts, which is a measured equilibrium, so they fold the way
// the joints actually bend rather than the way they look like they should.
// New templates are editable joint references, not trained skills. Range
// checks and finite open-loop playback do not establish balance, locomotion,
// takeoff, or a full-body roll; those require task-specific physics evidence.

import type { MotionFile } from "./format.ts";

export const BUILTIN_MOTIONS: MotionFile[] = [
  {
    format: 1,
    name: "nod",
    description: "Tip the beak down and back up. Head only — the body holds still.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: ["head_pitch"],
    keyframes: [
      { t: 0.0, pose: [0.3491] },
      { t: 0.5, pose: [-0.1] },
      { t: 1.0, pose: [0.3491] },
    ],
  },
  {
    format: 1,
    name: "look_around",
    description: "Sweep the head left and right, twice as slow as a nod.",
    loop: true,
    units: "deg",
    interp: "cubic",
    joints: ["head_yaw"],
    keyframes: [
      { t: 0.0, pose: [0] },
      { t: 0.9, pose: [40] },
      { t: 1.8, pose: [0] },
      { t: 2.7, pose: [-40] },
      { t: 3.6, pose: [0] },
    ],
  },
  {
    format: 1,
    name: "squat",
    description: "Fold halfway to a sit and rise again, keeping the trunk level.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: [
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 0.9, pose: [-0.0436, -0.4329, 0.6725, 0.2265, 0.0436, 0.4329, -0.6725, -0.2265] },
      { t: 1.8, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "bow",
    description: "Dip the head forward over a slight knee bend, hold, then rise.",
    loop: false,
    units: "rad",
    interp: "cubic",
    joints: [
      "neck_pitch", "head_pitch",
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [0.3491, 0.3491, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 1.0, pose: [0.9500, -0.0500, -0.0567, -0.4404, 0.4693, 0.2944, 0.0567, 0.4404, -0.4693, -0.2944] },
      { t: 2.0, pose: [0.9500, -0.0500, -0.0567, -0.4404, 0.4693, 0.2944, 0.0567, 0.4404, -0.4693, -0.2944] },
      { t: 3.0, pose: [0.3491, 0.3491, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "march_in_place",
    description: "Alternating leg-fold reference for a marching rhythm. Joint targets only; not a validated balancing or walking policy.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: ["left_hip_pitch", "left_knee", "left_ankle", "right_hip_pitch", "right_knee", "right_ankle"],
    keyframes: [
      { t: 0.0, pose: [-0.4579, -0.0049, 0.4530, 0.4579, 0.0049, -0.4530] },
      { t: 0.4, pose: [-0.6500, 0.5500, 0.1500, 0.4579, 0.0049, -0.4530] },
      { t: 0.8, pose: [-0.4579, -0.0049, 0.4530, 0.4579, 0.0049, -0.4530] },
      { t: 1.2, pose: [-0.4579, -0.0049, 0.4530, 0.6500, -0.5500, -0.1500] },
      { t: 1.6, pose: [-0.4579, -0.0049, 0.4530, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "walk_cycle_reference",
    description: "Opposed hip swings with alternating knee bends. A joint-space walking reference, not validated forward locomotion.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: ["left_hip_pitch", "left_knee", "left_ankle", "right_hip_pitch", "right_knee", "right_ankle"],
    keyframes: [
      { t: 0.0, pose: [-0.4579, -0.0049, 0.4530, 0.4579, 0.0049, -0.4530] },
      { t: 0.45, pose: [-0.6800, 0.3000, 0.3500, 0.2800, -0.0500, -0.4000] },
      { t: 0.9, pose: [-0.4579, -0.0049, 0.4530, 0.4579, 0.0049, -0.4530] },
      { t: 1.35, pose: [-0.2800, 0.0500, 0.4000, 0.6800, -0.3000, -0.3500] },
      { t: 1.8, pose: [-0.4579, -0.0049, 0.4530, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "sidestep_reference",
    description: "Alternate hip-roll offsets and bent-knee poses. No root translation is authored; this is not a validated sidestep gait.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: [
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 0.6, pose: [-0.2400, -0.5200, 0.3000, 0.2800, -0.0400, 0.4579, 0.0049, -0.4530] },
      { t: 1.2, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 1.8, pose: [0.0400, -0.4579, -0.0049, 0.4530, 0.2400, 0.5200, -0.3000, -0.2800] },
      { t: 2.4, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "crouch_hold",
    description: "Fold into a deeper crouch reference, hold, then return. Joint limits are checked; holding balance is a separate learned task.",
    loop: false,
    units: "rad",
    interp: "cubic",
    joints: [
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 0.8, pose: [-0.0300, -0.4200, 0.9500, 0.1400, 0.0300, 0.4200, -0.9500, -0.1400] },
      { t: 1.0, pose: [-0.0300, -0.4200, 0.9500, 0.1400, 0.0300, 0.4200, -0.9500, -0.1400] },
      { t: 2.2, pose: [-0.0300, -0.4200, 0.9500, 0.1400, 0.0300, 0.4200, -0.9500, -0.1400] },
      { t: 2.4, pose: [-0.0300, -0.4200, 0.9500, 0.1400, 0.0300, 0.4200, -0.9500, -0.1400] },
      { t: 3.2, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "jump_preparation",
    description: "Compress the legs then extend to the standing reference. Preparation targets only; no takeoff, airtime, or landing is implied.",
    loop: false,
    units: "rad",
    interp: "cubic",
    joints: [
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 0.7, pose: [-0.0350, -0.4250, 0.8500, 0.1800, 0.0350, 0.4250, -0.8500, -0.1800] },
      { t: 1.0, pose: [-0.0350, -0.4250, 0.8500, 0.1800, 0.0350, 0.4250, -0.8500, -0.1800] },
      { t: 1.3, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 1.9, pose: [-0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "tuck_preparation",
    description: "Fold both knees and lower the head into a compact reference pose. This does not encode a root rotation or a learned flip.",
    loop: false,
    units: "rad",
    interp: "cubic",
    joints: [
      "neck_pitch", "head_pitch",
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [0.3491, 0.3491, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 1.2, pose: [0.7000, -0.2500, -0.0200, -0.4100, 1.1000, 0.0800, 0.0200, 0.4100, -1.1000, -0.0800] },
      { t: 2.2, pose: [0.7000, -0.2500, -0.0200, -0.4100, 1.1000, 0.0800, 0.0200, 0.4100, -1.1000, -0.0800] },
      { t: 3.4, pose: [0.3491, 0.3491, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "roll_preparation",
    description: "A tucked reference with asymmetric hip and head-roll offsets. Joint-only preparation, not a full-body roll or flip.",
    loop: false,
    units: "rad",
    interp: "cubic",
    joints: [
      "neck_pitch", "head_pitch", "head_roll",
      "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
      "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ],
    keyframes: [
      { t: 0.0, pose: [0.3491, 0.3491, 0, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
      { t: 1.0, pose: [0.6500, -0.2000, 0, -0.0300, -0.4200, 0.9000, 0.1400, 0.0300, 0.4200, -0.9000, -0.1400] },
      { t: 1.7, pose: [0.6500, -0.2000, -0.2000, -0.2200, -0.4200, 1.0500, 0.1000, -0.0800, 0.4200, -0.8000, -0.1800] },
      { t: 2.4, pose: [0.6500, -0.2000, 0, -0.0300, -0.4200, 0.9000, 0.1400, 0.0300, 0.4200, -0.9000, -0.1400] },
      { t: 3.4, pose: [0.3491, 0.3491, 0, -0.0873, -0.4579, -0.0049, 0.4530, 0.0873, 0.4579, 0.0049, -0.4530] },
    ],
  },
  {
    format: 1,
    name: "dance_sway",
    description: "A gentle side-to-side joint reference with head turns and knee pulses. Balance during playback is not guaranteed.",
    loop: true,
    units: "rad",
    interp: "cubic",
    joints: ["head_yaw", "head_roll", "left_hip_roll", "left_knee", "right_hip_roll", "right_knee"],
    keyframes: [
      { t: 0.0, pose: [0, 0, -0.0873, -0.0049, 0.0873, 0.0049] },
      { t: 0.6, pose: [0.4500, 0.1800, 0.0400, 0.2000, 0.2100, -0.2000] },
      { t: 1.2, pose: [0, 0, -0.0873, -0.0049, 0.0873, 0.0049] },
      { t: 1.8, pose: [-0.4500, -0.1800, -0.2100, 0.2000, -0.0400, -0.2000] },
      { t: 2.4, pose: [0, 0, -0.0873, -0.0049, 0.0873, 0.0049] },
    ],
  },
  {
    format: 1,
    name: "neck_stretch",
    description: "Slow neck extension, head tilt, and return to the neutral reference. Undriven joints retain their standing targets.",
    loop: false,
    units: "rad",
    interp: "cubic",
    joints: ["neck_pitch", "head_pitch", "head_roll"],
    keyframes: [
      { t: 0.0, pose: [0.3491, 0.3491, 0] },
      { t: 1.0, pose: [-0.1500, 0.6500, 0] },
      { t: 2.0, pose: [0.1000, 0.4500, 0.2500] },
      { t: 3.0, pose: [0.1000, 0.4500, -0.2500] },
      { t: 4.0, pose: [0.3491, 0.3491, 0] },
    ],
  },
];

export function builtinMotionNames(): string[] {
  return BUILTIN_MOTIONS.map((m) => m.name ?? "untitled");
}
