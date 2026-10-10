# In-browser training — plan

Status: **M0, M1 and M2 done** — the Train workspace runs a throughput harness,
a vectorized environment, and PPO with checkpoint/resume.

Decisions taken (2026-09-02):

- **Browser-native first**, with the sim2real seams designed in from day one so
  BAM/DR is additive rather than a rewrite. See "The four seams".
- **Interactive loop first**, but checkpoint/resume is first-class from M2 so
  overnight runs work later without redesign.

## 1. The budget — measured

M0 ships a sweep in the Train workspace: each cell spins up a fresh worker
pool, allocates its environments, and steps them all at once. Numbers below are
from an M-series Mac (6 P-cores + 12 E-cores, `hardwareConcurrency` 18),
`robot_walk-nv`, 16 envs per worker, 2 s windows, Simulate paused.

| Workers | With policy forward | Physics only | Heap |
| ---: | ---: | ---: | ---: |
| 1 | 5,184 | 16,014 | 312 MB |
| 2 | 10,485 | 31,818 | 623 MB |
| 4 | 18,855 | 58,718 | 1,246 MB |
| 8 | 32,193 | 102,578 | 2,493 MB |
| 16 | **58,540** | 185,610 | 4,985 MB |
| 18 | 58,392 | **193,499** | 5,608 MB |

Control steps/s. Scaling is near-linear to 16 workers and then flat — the E-cores
stop contributing. Run-to-run variance is ±15%, so treat ~58 k as the working
figure, not 58,392.

**The plan assumed 40,000; the machine does ~58,000.** So:

| Target | Steps | Wall clock |
| --- | --- | --- |
| Full reference recipe (15,000 iters) | 1.47 B | **7 h** |
| Plausible convergence (2,000 iters) | 200 M | **56 min** |
| Bootstrap / debug (500 iters) | 50 M | **14 min** |

### What M0 overturned

**Physics is not the bottleneck — inference is.** Physics alone sustains
193,499 steps/s; adding a policy-shaped forward pass drops it to 58,392, so the
naive scalar-JS MLP eats **70% of the budget**. That is ~3× of headroom sitting
in a single function, and it re-prioritises M3: the rollout's inference path
matters far more than the learner's.

At the physics-only ceiling the full reference recipe would take 2.1 h — the
CUDA box's own number. Inference will never be free, but WASM SIMD or a batched
WebGPU pass should recover a good part of that gap.

**Visual geoms cost memory, not time.** Every model now ships a `-nv` twin with
`class="visual"` geoms and the meshes only they referenced removed —
`robot_walk-nv` is 6 geoms and 4 meshes against 76 and 38. Because visual geoms
are already `contype=0 conaffinity=0`, this buys only ~7% throughput, but it
cuts per-worker heap from 746 MB to 312 MB and makes a pool start much faster.
`prepare-assets` compiles both twins and asserts `nq`/`nv`/`nu`/`nbody` and
total mass match, so a bad strip fails the build rather than the policy.

**Memory, not CPU, caps the worker count.** 18 workers × 16 envs is 5.6 GB.
`MjData` is ~14 MB regardless of model, so envs-per-worker is the expensive
axis and worker count is the cheap one.

**Bigger per-worker batches do not help.** 64 envs/worker measured slightly
*worse* than 16 (4,949 vs 5,150 per thread). The earlier Node result suggesting
otherwise was on the un-stripped model, where collision cost dominated.

**Worker count saturates at the P-core count × ~2.7.** 16 and 18 workers tie.
Defaulting the pool to `hardwareConcurrency` is fine but not free — the last
few workers cost memory and return nothing.

**Model choice barely moves throughput.** At 8 workers: `walk-nv` 31.6k,
`groundcontact-nv` 28.0k, `allcollisions-nv` 23.6k. Worth picking the light one,
not worth agonising over.

## 2. Architecture

Five layers, each independently testable:

1. **Vectorized env** — `N` workers × `K` envs. Each worker owns one `MjModel`,
   `K` `MjData`, and does reset / DR / obs / reward / termination locally,
   returning packed `Float32Array`s by transfer. No `SharedArrayBuffer`, so no
   COOP/COEP requirement on the host.
2. **Rollout inference** — the policy runs *inside* each worker as a `[K×61]`
   GEMM, not a per-env GEMV. Batched, and no cross-thread barrier per control
   step.
3. **Env spec** — a declarative term list (reward / termination / event /
   curriculum) mirroring mjlab's manager pattern, so terms are composable and
   unit-testable one at a time.
4. **Learner** — hand-rolled MLP + backprop + Adam + GAE + clipped surrogate.
   The architecture is fixed and tiny (512/256/128, ~197 k params); a framework
   would cost megabytes for generality we do not need. CPU first, WebGPU once
   it is the measured bottleneck.
5. **UI** — run config, live reward/loss charts, a live-rendered training env,
   checkpoint list, ONNX export landing straight in Simulate.

**The training loop must live in workers, not `requestAnimationFrame`.**
Background tabs throttle rAF to ~1 Hz, which would silently stall an overnight
run. The main thread only renders and charts.

## 3. The four seams

Browser-native today, sim2real later, without a rewrite. Each of these is an
interface from the first commit, with a trivial implementation now:

| Seam | Now (browser-native) | Later (sim2real) |
| --- | --- | --- |
| `Actuator` | `XmlPositionActuator` — write targets to `data.ctrl`, MJCF `position` does the PD (kp 0.55) | `BamActuator`: the M6 voltage model + fitted `xl330` parameters, from [Rhoban/bam](https://github.com/Rhoban/bam) |
| `DelayedActuator` | lag 0 | lag 3–6 control steps (`_BAM_ACTUATOR_KWARGS`) — cheap, so build it now |
| `Randomizer` registry | empty | CoM, head CoM, mass/inertia, armature, joint friction, encoder bias, IMU misalignment, velocity pushes |
| `ObsPipeline` | raw | + per-term noise, encoder bias, IMU misalignment |

Two rules the interfaces must enforce, both learned the hard way upstream (see
`microduck_rl/AGENTS.md`):

- **Randomizers restore-then-apply.** An accumulating CoM randomizer silently
  degraded every long run for months. Make non-accumulation the interface's
  contract, not each implementation's responsibility.
- **Joint indices resolve by name, never by position.** Backlash and roller
  models interleave `passive_*` joints into `qpos`. `src/sim/controller.ts`
  already does this; the env spec must too.

Model choice (`walk` / `groundcontact` / `allcollisions` / `*_backlash`) is
config, never hardcoded.

Sizing the sim2real port: BAM is one `compute()` (a voltage-controlled DC motor
with a Coulomb + Stribeck + load-dependent friction budget) plus a parameter
block. `friction_dr_bam.py` in the submodule is a ~112-line subclass; the model
itself lives in the `bam` package. Bounded, but not free.

## 4. Verification — before any training

RL implementations fail silently and slowly: you burn an hour of compute to
learn nothing. Each step below gates the next.

1. **Replay `alpha_stand` through our env and check the reward curve.** We
   already have a known-good policy. If our ported reward stack does not score
   it highly, the reward stack is wrong. Cheap, and nothing else gives that
   signal.
2. **Gradient-check** backprop against finite differences.
3. **Solve a toy task** (point-mass reach) end to end. If PPO cannot do that in
   seconds, it will not do standup in an hour.
4. **Bootstrap task: "hold the STAND pose"** from a noisy init. Trivially
   learnable — the hello-world that proves the full loop.
5. **Only then** the real sit→stand reward stack.

### M1 result

`npm run check:env` replays three controllers through the same `VecEnv` with
the same seed, so they see identical reset poses:

| Controller | Reward / step | Standing | Mean trunk z |
| --- | ---: | ---: | ---: |
| `alpha_stand` | **8.911** | 82.3% | 10.8 cm |
| zero (hold reference pose) | 1.644 | 0.0% | 5.2 cm |
| random | 1.247 | 0.0% | 5.2 cm |

5.4× separation, every self-negating penalty came back negative, and the
`standing_composite` product carries the largest share of the positive mass —
which is what the reference intends. The ported reward stack recognises good
behaviour.

Node 24 strips TypeScript natively, so that check imports the real `src/`
modules rather than a reimplementation — the whole point of the gate. It
needed two concessions: explicit `.ts` import extensions repo-wide, and no
TypeScript parameter properties (strip-only mode rejects them).

One thing the replay turned up: **the exported ONNX policies have a fixed batch
dimension of 1.** They were traced for single-robot deployment, so a vectorized
replay has to call them once per environment. This costs the gate, not the
trainer — M2's learner uses its own MLP with no such limit — but it rules out
using an exported checkpoint directly as a fast rollout policy, which matters
for M4 fine-tuning.

### M2 result

Four headless gates, all in `npm run check`:

| Gate | Result |
| --- | --- |
| `check:grad` — finite-difference the backprop | norm-relative error 1.5e-5 |
| `check:ppo` — solve a toy reach task | mean distance 2.248 → 0.093 |
| `check:trainer` — learn to hold the pose | reward 3.64 → 5.72, standing 22.7% → 51.3% |
| `check:trainer` — checkpoint round trip | drift 0.00e+0 (exact) |

**"Hold the pose" is not a freebie.** With `ctrl` frozen at the reference pose,
the duck holds for ~0.5 s and then topples: STAND is not a passive equilibrium.
The reference's `STAND_Z = 0.115` was *measured under a policy actively
balancing*, not at rest. So the bootstrap task is genuine active balance, and
the untrained baseline is ~22% standing rather than 100%. This is exactly the
check AGENTS.md prescribes before training, and it would have been an
expensive assumption to carry into a long run.

### Where a training iteration actually goes

`npm run profile` breaks an iteration into its parts, at 32 envs × 24 steps,
5 epochs × 4 minibatches of 192:

| | 128/64 | 512/256/128 (reference) |
| --- | ---: | ---: |
| Rollout physics + reward | 106 ms (34%) | 85 ms (**3.4%**) |
| Rollout policy forward | 18 ms (6%) | 197 ms (7.8%) |
| PPO update | 207 ms (66%) | 2,248 ms (**88.9%**) |
| — of which forward | 86 ms | 993 ms |
| — of which backward + clip + Adam | 120 ms | 1,257 ms |
| — of which everything else | 1 ms | **1 ms** |
| **Dense linear algebra** | 71% | **96.7%** |

**This overturns what M0 implied about M3.** M0 measured a rollout-only loop
and found inference at ~70% of the step budget. With a learner attached that
is no longer the target: rollout inference is 7.8%, and the update's own
forward/backward is 89%. Each sample is forwarded once during rollout but
forwarded *and* backwarded five times during the update — so the update costs
~11× the rollout's inference, and optimising rollout inference first would
have chased 8% of the problem.

Physics is 3.4%. It is not worth touching.

Two cheap levers exist before any SIMD work: `epochs` (5 is the reference;
2 would cut the update 2.5× at some sample-efficiency cost) and minibatch size,
which a vectorised kernel would amortise better.

A note on measuring this: `MlpNet.backward` skips a row whose incoming
gradient is exactly zero, and the PPO clip fraction decides how many those are
(30–42% here). Microbenchmarks with zero-filled gradients therefore price the
backward pass at nearly nothing, and ones with dense synthetic gradients
over-price it by ~28%. `UpdateStats` now reports its own `fwdMs`/`bwdMs`/
`otherMs`, measured in place, because neither synthetic bracket was right.

### M3 result — 7.5× on a training iteration

At the reference net, 32 envs × 24 steps:

| | before | after |
| --- | ---: | ---: |
| Rollout physics + reward | 85 ms | 86 ms |
| Rollout policy forward | 197 ms | 15 ms |
| Update forward | 993 ms | 90 ms |
| Update backward | 1,236 ms | 127 ms |
| Clip + Adam | 21 ms | 21 ms |
| **Per iteration** | **2,530 ms** | **339 ms** |

Two steps. First, blocking the JavaScript kernels — eight samples share each
weight-row load instead of re-streaming the whole matrix per sample — for
1.75×, with forward results *bit-identical* because each accumulator still
sums in the same order. Then SIMD kernels in WebAssembly (`src/train/kernels/`)
for another 4.3×; `check:kernels` asserts they match the JavaScript reference
to ~2e-7 relative and the JavaScript path stays as both reference and fallback.

**The bottleneck flipped.** Physics became 25% of an iteration at the reference
net and **74%** at the small one, while the learner dropped to 4–6%. So the
next speedup was not a faster learner but more rollout workers feeding one —
now built (`RolloutSource`, with an in-thread and a worker-pool
implementation). Measured in Chrome at 32 envs per worker, small net:

| Rollout workers | Envs | Iteration | Control steps/s | Rollout / update |
| ---: | ---: | ---: | ---: | ---: |
| 1 (in-thread) | 32 | 125 ms | 6,144 | 82% / 18% |
| 2 | 64 | 161 ms | 9,540 | 73% / 27% |
| 4 | 128 | 209 ms | 14,699 | 57% / 43% |

2.4× throughput at 4 workers, and the split is heading back toward the
learner — which is now the thing to watch again.

Two things broke on the way, both from the kernel heap being a bump allocator
that never frees: `#ensureBatch` reallocated whenever the batch size changed,
and training alternates rollout (32) and minibatch (192) every iteration, so it
leaked until the heap ran out. It is grow-only now. And `kalloc` bumps a
pointer without growing the memory behind it — an overrun there would have been
a wild pointer rather than an error.

Also worth recording: the hand-written SIMD feature probe I used first was
malformed and reported "no SIMD" on an engine that supports it. Validating the
real module is both simpler and tests the thing that actually matters.

### Three bugs that only a gate would have caught

**The observation normalizer must not update between rollout and update.**
PPO's importance ratio assumes the stored log-probs came from the same function
the first minibatch evaluates. Re-scaling the inputs in between produced a
phantom KL of 0.82 on iteration one with frozen weights, which drove the
adaptive learning rate to its floor and froze the run — while every individual
component looked correct. `UpdateStats.initialKl` now asserts the invariant
directly: it is ~0 (3.9e-14) when rollout and update agree.

**Environments must be staggered.** Created together, all of them reset on the
same step forever, so each rollout sampled one narrow slice of episode time
instead of the state distribution. The metrics oscillated with the episode
period and looked like wild instability. `VecEnv.stagger()` fixes it — and
`evaluate()` has to re-stagger afterwards, because an aligned evaluation reset
silently re-introduces it.

**KL scales as (Δmean / σ)².** Actions are joint offsets in radians, so the
reference `init_std = 1.0` is ~57° of noise per joint per control step: the
duck is shaken apart and never experiences standing, leaving PPO nothing to
reinforce. But at a sane σ = 0.1 the reference `desired_kl = 0.01` — tuned at
σ = 1.0 — pins the learning rate to its floor instead. Both constants had to
move together.

## 5. Checkpoint / resume

Implemented in M2. The checkpoint covers policy + critic parameters, Adam
moments, obs-normalizer running mean/var, iteration counter and RNG state —
anything whose loss would make a resumed run differ from an uninterrupted one.
`check:trainer` asserts a restored trainer evaluates bit-identically.

One subtlety worth keeping: the RNG's state lives in a field, not a closure.
`VecEnv` and `ActorCritic` capture the generator *function* at construction, so
swapping in a fresh closure on restore would leave them drawing from the old
stream and the resume would silently diverge.

Persisted to OPFS (megabytes of weights, and the API is a plain file handle),
autosaving every 25 iterations so a run survives a reload. Download/upload for
moving a run between machines is still to do.

## 6. Milestones

| | Goal | Gate |
| --- | --- | --- |
| ~~M0~~ | ~~Throughput harness~~ | ✅ ~58 k steps/s — section 1 |
| ~~M1~~ | ~~Vectorized env + the four seams~~ | ✅ `alpha_stand` 8.91 vs 1.64 do-nothing — section 4 |
| ~~M2~~ | ~~PPO on CPU + checkpoint/resume~~ | ✅ toy task solved, hold-pose learned — section 4 |
| ~~M3~~ | ~~The learner's forward/backward~~ | ✅ 7.5× — 2,530 ms → 339 ms per iteration |
| **M4** | Fine-tune from a shipped checkpoint | A visibly adapted policy |
| **M5** | From-scratch standup, ONNX export, round-trip into Simulate | A policy we trained, running in the Simulate tab |

**M4 before M5 on purpose.** Fine-tuning converges in far fewer steps and is
the most likely thing to actually work in a tab. M0 came in above estimate, so
M5 now looks reachable rather than aspirational — but the ordering stands,
because a 56-minute run is still a bad debugging loop.

## 7. Known caveats

- **Browser-trained ≠ sim2real-ready** until the seams are filled. A policy
  trained against MJCF `position` actuators is tuned to *our* actuator model,
  not the real servo. That `alpha_stand` transfers into our sim is a nice
  robustness result, not evidence the reverse direction works.
- **MuJoCo WASM (CPU, f64) ≠ MuJoCo Warp (GPU, f32).** Fine for browser-only
  use; another reason not to promise hardware transfer.
- **The reward playbook in `AGENTS.md` encodes 15,000-iteration lessons.** At
  small scale we will rediscover reward hacking on our own; budget for a few
  rounds of whack-a-mole.
- **WebGPU availability** gates M3. Needs a CPU fallback or an explicit
  capability gate in the UI.
