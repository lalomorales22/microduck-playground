<script lang="ts">
  import type { View } from "./views.ts";

  const { go }: { go: (v: View) => void } = $props();

  const capabilities = [
    {
      state: "ready",
      title: "Run a trained policy",
      body: "MuJoCo in WebAssembly, ONNX inference, 50 Hz. Knock the duck over and watch it get up.",
      action: "Open Simulate",
      view: "sim" as View,
    },
    {
      state: "ready",
      title: "Train one yourself",
      body: "PPO over a vectorized environment in a worker. Checkpoints survive a reload.",
      action: "Open Train",
      view: "train" as View,
    },
    {
      state: "ready",
      title: "Watch what you trained",
      body: "Save a run, then pick it as the driving policy in Simulate. Download it from Files to keep it.",
      action: "Open Files",
      view: "files" as View,
    },
    {
      state: "ready",
      title: "Show it a motion",
      body: "Keyframe a pose over time — or have an AI agent write the file — and train a policy to perform it while balancing.",
      action: "Open Motion maker",
      view: "motion" as View,
    },
    {
      state: "soon",
      title: "Export and deploy",
      body: "Send a trained policy to a real duck. Needs ONNX export and the BAM actuator model first.",
      action: null,
      view: null,
    },
  ];

  const steps = [
    {
      n: 1,
      title: "Say what the skill is, as a reward",
      where: "src/train/env/rewards.ts",
      body: "One term per thing you care about. For one-footed standing: keep everything that already rewards standing, and add clearance for the lifted foot.",
      code: `{
  name: "right_foot_lift",
  weight: 2.0,
  compute: (ctx) => {
    // Foot bodies are ankle_left / ankle_right.
    const z = ctx.data.body("ankle_right").xpos[2];
    return Math.min(z / 0.05, 1);   // pays up to 5 cm, then stops
  },
}`,
    },
    {
      n: 2,
      title: "Say how an episode starts",
      where: "src/train/env/standup.ts",
      body: "Standing on two feet is the right start for this one — the duck has to discover the shift, not the stand. resetMix: { stand: 1 }.",
      code: null,
    },
    {
      n: 3,
      title: "Check your reward before you train on it",
      where: "npm run check:env",
      body: "Replays the shipped policy, a do-nothing baseline and random noise through your reward. If it does not rank them correctly, no amount of training will fix it.",
      code: null,
    },
    {
      n: 4,
      title: "Check the pose is physically reachable",
      where: "a 3-second sim",
      body: "Hold the target pose open-loop and watch the tilt, not just the height. A target the robot cannot hold is a target it will never reach — and this repo already learned that the two-footed stand is not a passive equilibrium either.",
      code: null,
    },
    {
      n: 5,
      title: "Train",
      where: "the Train workspace",
      body: "Start small: 32 envs, the 128/64 net, a few hundred iterations. Watch the standing line, not the reward line — reward can climb while the behaviour gets worse.",
      code: null,
    },
    {
      n: 6,
      title: "Expect to be cheated, twice",
      where: "",
      body: "The first policy will find something that scores well and looks nothing like the skill. Hopping on one foot. Leaning on the raised leg. That is normal; tighten the reward and go again.",
      code: null,
    },
    {
      n: 7,
      title: "Watch it",
      where: "Save as… → Simulate",
      body: "Save the run under a name, then pick it as the driving policy in Simulate. Reading a reward curve is not the same as seeing what the policy actually does.",
      code: null,
    },
  ];

  /** The other route: some skills are easier to demonstrate than to describe. */
  const motionSteps = [
    {
      n: 1,
      title: "Keyframe it, or have it written",
      where: "the Motion maker",
      body: "Drag the joint sliders at a few points on the timeline and the keyframes appear as you go. Only the joints you touch become part of the motion; the rest hold the reference pose. Or hand docs/motion-format.md to an AI agent with \u201cmake the duck take a slow bow\u201d and paste what it writes into the JSON panel — a wrong angle is visible, which a wrong reward is not.",
      code: null,
    },
    {
      n: 2,
      title: "Watch it before you train on it",
      where: "Preview \u2194 Physics",
      body: "Preview places the joints exactly as written, with no physics — the view for checking it looks the way you meant. Switch to physics playback and it will fall over: open-loop, this robot topples in about a second whatever you ask of it. That is the task, not a bug in your file.",
      code: null,
    },
    {
      n: 3,
      title: "Train on it",
      where: "Train \u2192 Task \u2192 your motion",
      body: "The policy is scored on matching your angles while staying on its feet. It is told only where in the motion it is, so what you get back is a self-contained policy that carries the motion itself.",
      code: null,
    },
    {
      n: 4,
      title: "Expect some motions to be impossible",
      where: "",
      body: "The duck's head is ~38% of its mass. A pose that looks fine in a file may not be reachable while balancing, and the failure looks like a training bug. Keep the mass over the feet.",
      code: null,
    },
  ];

  const rules = [
    ["It optimises the letter of the reward", "Every degree of freedom you leave unspecified gets exploited. Encode what counts as the skill in hard state checks, not small nudges."],
    ["No jackpots", "A \"reach X\" bonus that then pays per step buys arbitrary violence to get there early. Rate-limit it."],
    ["Never pay for being in a bad state", "Reward a fallen duck for anything and it will park there and farm it."],
    ["Penalties must read negative", "Every penalty term should be ≤ 0 in the breakdown. A sign slip turns it into a reward for the violation."],
  ];
</script>

<div class="guide">
  <header>
    <h2>Train a robot duck in your browser</h2>
    <p>
      Physics, policy and learning all run in this tab — no server, no CUDA, no
      Python. The robot is <a href="https://github.com/pollen-robotics/microduck" target="_blank" rel="noreferrer">Microduck</a>,
      ~800 g and ~25 cm of bipedal trouble.
    </p>
  </header>

  <section class="cards">
    {#each capabilities as c (c.title)}
      <article class:soon={c.state === "soon"}>
        <span class="pill">{c.state === "ready" ? "Ready" : "Not yet"}</span>
        <h3>{c.title}</h3>
        <p>{c.body}</p>
        {#if c.action && c.view}
          <button onclick={() => go(c.view)}>{c.action} →</button>
        {/if}
      </article>
    {/each}
  </section>

  <section>
    <h3 class="section">Teach it something new</h3>
    <p class="lede">
      Worked example: <strong>stand on one foot</strong>. Steps 1 and 2 are code
      edits today — the reward lives in TypeScript, not in a form. Everything
      after that is a command or a button.
    </p>

    <ol class="steps">
      {#each steps as s (s.n)}
        <li>
          <span class="n">{s.n}</span>
          <div>
            <h4>{s.title} {#if s.where}<code>{s.where}</code>{/if}</h4>
            <p>{s.body}</p>
            {#if s.code}<pre>{s.code}</pre>{/if}
          </div>
        </li>
      {/each}
    </ol>
  </section>

  <section>
    <h3 class="section">Or show it, instead of describing it</h3>
    <p class="lede">
      Reward engineering is the right tool for a goal defined by an outcome —
      get up, don't fall, hold a velocity. For anything with a <em>shape</em> —
      a bow, a wave, a dance — it is easier to demonstrate the skill than to
      write a function that scores it.
    </p>

    <ol class="steps">
      {#each motionSteps as s (s.n)}
        <li>
          <span class="n">{s.n}</span>
          <div>
            <h4>{s.title} {#if s.where}<code>{s.where}</code>{/if}</h4>
            <p>{s.body}</p>
          </div>
        </li>
      {/each}
    </ol>
  </section>

  <section>
    <h3 class="section">Four rules that save the most time</h3>
    <p class="lede">
      Distilled from the reference project's playbook — each one is there
      because a run failed that way.
    </p>
    <dl class="rules">
      {#each rules as [title, body] (title)}
        <div><dt>{title}</dt><dd>{body}</dd></div>
      {/each}
    </dl>
  </section>

  <footer>
    <span>Next up: authoring skills without touching code — see
      <code>docs/training-plan.md</code>.</span>
  </footer>
</div>

<style>
  .guide {
    grid-area: main;
    padding: 22px 24px 32px; overflow-y: auto;
    background: var(--viewport);
    display: flex; flex-direction: column; gap: 26px;
  }

  header h2 { margin: 0; font-size: 20px; letter-spacing: 0; }
  header p { margin: 6px 0 0; max-width: 62ch; }
  p { font-size: 12.5px; line-height: 1.6; color: var(--muted); margin: 0; }
  a { color: var(--accent-text); text-decoration: none; }
  a:hover { text-decoration: underline; }
  code {
    background: var(--panel-hi); padding: 1px 5px; border-radius: 4px;
    font-size: 11px; color: var(--ink);
  }

  h3.section { margin: 0 0 6px; font-size: 14px; letter-spacing: 0; }
  .lede { max-width: 68ch; margin-bottom: 14px; }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
  article {
    display: flex; flex-direction: column; gap: 7px; align-items: flex-start;
    padding: 14px; border: 1px solid var(--line); border-radius: 10px;
    background: var(--panel);
  }
  article.soon { color: var(--muted); }
  article h3 { margin: 0; font-size: 13px; }
  .pill {
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0; padding: 2px 7px; border-radius: 999px;
    border: 1px solid var(--line-hi); color: var(--ok);
  }
  article.soon .pill { color: var(--warn); }
  article button {
    margin-top: auto; font: inherit; font-size: 11px; font-weight: 600;
    color: var(--accent-text); background: none; border: 0; padding: 4px 0; cursor: pointer;
  }
  article button:hover { text-decoration: underline; }

  .steps { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 14px; }
  .steps li { display: flex; gap: 12px; }
  .steps > li > div { min-width: 0; flex: 1; }
  .n {
    flex: none; display: grid; place-items: center;
    width: 21px; height: 21px; border-radius: 50%;
    background: var(--panel-hi); border: 1px solid var(--line);
    font-size: 11px; font-weight: 700; color: var(--accent-text);
  }
  h4 { margin: 2px 0 4px; font-size: 12.5px; font-weight: 600; }
  pre {
    margin: 8px 0 0; padding: 10px 12px; overflow-x: auto;
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    font-size: 11px; line-height: 1.5; color: var(--ink);
  }

  .rules { margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
  .rules div {
    padding: 11px 13px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel);
  }
  dt { font-size: 12px; font-weight: 600; margin-bottom: 3px; }
  dd { margin: 0; font-size: 11.5px; line-height: 1.55; color: var(--muted); }

  footer { padding-top: 14px; border-top: 1px solid var(--line); font-size: 11px; color: var(--muted); }
</style>
