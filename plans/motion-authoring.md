# Motion authoring — teach a skill by showing it

Status: **stage 1 is built.** The motion file, the interpolator, the imitation
reward and the viewport preview all ship — see `docs/motion-format.md`,
`src/motion/` and `src/train/env/imitation.ts`. Stage 2 (the keyframe editor)
is still an idea.

Two things stage 1 taught, worth knowing before stage 2:

- **Every motion is a balance problem.** Driven open-loop the duck topples in
  about a second whatever it is asked to do — even a head-only nod — because
  standing is an active behaviour on this robot, not a pose. The editor's
  physics playback will therefore always end in a fall, and that has to be
  presented as expected rather than as a verdict on the motion.
- **Score only the joints the file drives.** Averaging tracking error over all
  fourteen buries a one-joint nod under thirteen terms that match regardless.

---

Today a new skill means writing a reward function in TypeScript and then
fighting the policy for it. That is the hard way, and the reference project's
playbook is mostly a list of ways it goes wrong. Both ideas below replace
"describe the skill as a reward" with "show the skill", which sidesteps most of
the reward-hacking problem.

They are the same feature at two levels of ambition, and they share one
artifact: **a motion file**. Do them in this order — the format is the risky
part, and the second idea proves it out before any UI is built on top.

## Stage 1 — a motion file an agent can write

Define a JSON format for a pose or a motion: keyframes of 14 joint angles plus
timing, and ship a spec document alongside it. Then anyone can hand the spec to
an AI agent with "make the duck do a slow bow" and get a valid file back — no
editor, no UI work.

```jsonc
{
  "name": "bow",
  "loop": false,
  "joints": ["left_hip_yaw", "left_hip_roll", "..."],   // the 14, in policy order
  "keyframes": [
    { "t": 0.0, "pose": [0, -0.087, -0.458, "..."] },
    { "t": 1.2, "pose": [0, -0.087, -0.900, "..."] },
    { "t": 2.4, "pose": [0, -0.087, -0.458, "..."] }
  ]
}
```

Why this first:

- It is the same file the editor in stage 2 would produce, so the format gets
  stress-tested before any UI depends on it.
- It is useful on its own — a library of authored motions is worth having.
- An agent writing joint angles is a far easier ask than an agent writing a
  reward function, because a wrong pose is *visible* and a wrong reward is not.

Work: the schema, an interpolator (joint angles slerp badly — interpolate in
joint space, not Cartesian), a preview in the Simulate viewport, and the spec
document.

## Stage 2 — a keyframe editor

A timeline UI over the same format: scrub, set a pose by dragging joints,
key it, interpolate between. Closer to After Effects than to a robotics tool,
which is the point.

Work: pose editing (inverse kinematics would be nicer than per-joint sliders
but is a project of its own), a timeline, and playback against live physics so
you can see the motion fail before you train on it.

## What actually makes this work: the imitation reward

Both stages are worthless without the piece that turns a motion into training
signal. This is well-trodden ground — DeepMimic-style reference-motion tracking
— and the reward is roughly:

```
reward = w_pose * exp(-k * ||q - q_ref(phase)||^2)
       + w_vel  * exp(-k * ||qdot - qdot_ref(phase)||^2)
       + w_root * (upright / height terms already in rewards.ts)
```

with `phase` advancing with the episode clock and fed into the policy's command
slots — the 13D block already reserved for exactly this kind of thing.

Two difficulties worth knowing before starting:

- **An authored motion can be dynamically impossible.** The duck's head is
  ~38% of its body mass; a motion that looks right in an editor may not be
  physically achievable, and the policy will fail in ways that look like a
  training bug. The editor should play the motion against live physics, not
  just render it.
- **Phase alignment.** A policy that performs the motion correctly but slightly
  late is punished hard by a naive per-step tracking reward. The usual fixes
  are a phase-tolerant reward or letting the policy control its own phase
  advance.

## Verdict

Feasible, and a better fit than reward engineering for anything with a *shape*
— a bow, a wave, a dance, a run. Reward engineering stays the right tool for
goals defined by an outcome rather than a trajectory (get up, don't fall,
track a velocity).

Sequence: motion format + spec → imitation reward term → editor.
