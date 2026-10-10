// Official pretrained exports, pinned to the reviewed Hugging Face snapshot.
// Hashes and sizes come from its tree API (LFS SHA-256), not a moving branch.
// https://huggingface.co/pollen-robotics/microduck-policies/tree/6677ec007142bbaf05123a468f6ae58d13808786
export const POLICY_REPOSITORY = "pollen-robotics/microduck-policies";
export const POLICY_REVISION = "6677ec007142bbaf05123a468f6ae58d13808786";

export interface ShippedPolicy {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  /** Accepts body-frame [vx m/s, vy m/s, yaw rate rad/s] in command[0:3]. */
  readonly commands: boolean;
  readonly experimental?: boolean;
  readonly sha256: string;
  readonly bytes: number;
}

// These exports use obs[1,61] -> actions[1,14], with their normalizer baked in:
// gyro3, projected-gravity3, qpos-default14, qvel14, previous-action14,
// command13 (twist3, head4, body6). Targets are STAND2 + action * 1.0 at 50 Hz.
// This selector only exposes twist; the other ten commands stay zero.
// Sitstand overloads twist.vx as a posture flag; kicks/roulade need timed
// execution; roller needs different physics and scale 0.8. They are excluded.
export const SHIPPED_POLICIES: readonly ShippedPolicy[] = [
  {
    id: "alpha_stand.onnx",
    label: "Stand / recover",
    description: "Standing and get-up policy",
    commands: false,
    sha256: "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
    bytes: 793705,
  },
  {
    id: "velstand.onnx",
    label: "Walk / stand",
    description: "Walking + standing; low commands may remain stationary",
    commands: true,
    sha256: "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a",
    bytes: 793705,
  },
  {
    id: "alpha_walking.onnx",
    label: "Walk (legacy)",
    description: "Earlier velocity-controlled walking policy",
    commands: true,
    experimental: true,
    sha256: "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
    bytes: 793705,
  },
];
