// Node runs .ts by stripping types, which rejects a few TypeScript features.
// Every module under src/ must stay Node-loadable so the headless checks can
// import the real code instead of a copy — this guards that property.
import fs from "node:fs";
import path from "node:path";

const roots = ["src", "scripts"];
const banned = [
  {
    // `constructor(private x: T)` — the one that keeps biting.
    re: /constructor\s*\(([^)]*)\)/gs,
    test: (args) => /\b(private|public|protected|readonly)\s+\w+\s*:/.test(args),
    what: "parameter property in a constructor",
  },
  { re: /^\s*(export\s+)?enum\s+\w+/gm, test: () => true, what: "enum" },
  { re: /^\s*(export\s+)?namespace\s+\w+/gm, test: () => true, what: "namespace" },
];

const walk = (dir) =>
  fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? walk(p) : p.endsWith(".ts") ? [p] : [];
  });

const problems = [];
for (const root of roots) {
  if (!fs.existsSync(root)) continue;
  for (const file of walk(root)) {
    const src = fs.readFileSync(file, "utf8");
    for (const { re, test, what } of banned) {
      re.lastIndex = 0;
      for (let m; (m = re.exec(src)); ) {
        if (test(m[1] ?? "")) {
          problems.push(`${file}: ${what} — not supported by Node's strip-only TypeScript`);
        }
      }
    }
    if (/from\s+"\.[^"]*"/.test(src)) {
      for (const m of src.matchAll(/from\s+"(\.[^"]*)"/g)) {
        if (!/\.(ts|js|svelte|css|json)$/.test(m[1])) {
          problems.push(`${file}: relative import "${m[1]}" needs an explicit extension`);
        }
      }
    }
  }
}

if (problems.length) {
  console.error("Node compatibility problems:\n" + problems.map((p) => `  - ${p}`).join("\n"));
  process.exit(1);
}
console.log("node compat: ok");
