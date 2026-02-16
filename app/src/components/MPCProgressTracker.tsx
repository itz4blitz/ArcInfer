"use client";

import {
  Brain,
  Shrink,
  ShieldCheck,
  Send,
  Network,
  CircleCheck,
  Lock,
  type LucideIcon,
} from "lucide-react";

import type { MpcStatus } from "@/lib/program";

export type InferenceStage =
  | "idle"
  | "embedding"
  | "pca"
  | "encrypting"
  | "submitting"
  | "mpc"
  | "complete"
  | "error";

interface MPCProgressTrackerProps {
  stage: InferenceStage;
  error?: string;
  /** Arcium computation sub-status, shown during the MPC stage. */
  mpcStatus?: MpcStatus;
}

const MPC_STATUS_LABELS: Record<MpcStatus, string> = {
  polling: "Waiting for MPC nodes to pick up computation...",
  queued: "Queued in Arcium mempool, waiting for cluster assignment...",
  executing: "MPC cluster is running encrypted inference...",
  finalized: "Computation finalized, waiting for callback transaction...",
  failed: "MPC computation failed. You may retry.",
  unknown: "MPC nodes analyzing encrypted data",
};

interface StageInfo {
  key: InferenceStage;
  label: string;
  description: string;
  technical: string;
  privacy: string;
  icon: LucideIcon;
  privacyIcon: LucideIcon;
}

const STAGES: StageInfo[] = [
  {
    key: "embedding",
    label: "Tokenize & Embed",
    description: "Text is converted into a numerical representation using a neural network that runs entirely in your browser.",
    technical: "all-MiniLM-L6-v2 (ONNX) \u2192 384-dim sentence vector",
    privacy: "Runs locally \u2014 no data leaves your device",
    icon: Brain,
    privacyIcon: Lock,
  },
  {
    key: "pca",
    label: "Compress & Quantize",
    description: "The 384-dimensional embedding is reduced to 16 dimensions via PCA, then converted to fixed-point integers.",
    technical: "PCA (384\u219216) \u2192 Q16.16 fixed-point \u2192 16 integers",
    privacy: "Still local \u2014 dimensionality reduction happens in-browser",
    icon: Shrink,
    privacyIcon: Lock,
  },
  {
    key: "encrypting",
    label: "Encrypt",
    description: "Each of the 16 quantized values is encrypted using a shared secret derived from x25519 key exchange with the MPC cluster.",
    technical: "x25519 + RescueCipher \u2192 16 \u00d7 [u8; 32] ciphertexts",
    privacy: "Ciphertexts are indistinguishable from random \u2014 nobody can read them",
    icon: ShieldCheck,
    privacyIcon: ShieldCheck,
  },
  {
    key: "submitting",
    label: "Submit to Solana",
    description: "A Solana transaction carries the encrypted payload on-chain and queues MPC computation with the Arcium cluster.",
    technical: "classify_reveal IX \u2192 queue MPC via CPI \u2192 computation account created",
    privacy: "Only ciphertext + your public key touch the blockchain",
    icon: Send,
    privacyIcon: Lock,
  },
  {
    key: "mpc",
    label: "MPC Inference",
    description: "Arcium\u2019s MPC nodes secret-share the encrypted data and evaluate the neural network collaboratively \u2014 no single node ever sees the input.",
    technical: "16\u219216\u21928\u21922 NN, x\u00B2 activation, 426 params, ~10 MPC rounds",
    privacy: "Computed on secret-shared fragments across multiple nodes",
    icon: Network,
    privacyIcon: Network,
  },
  {
    key: "complete",
    label: "Result",
    description: "The MPC cluster emits the classification result (positive or negative) via a Solana callback transaction event.",
    technical: "Callback TX \u2192 ClassificationRevealedEvent(class=0|1)",
    privacy: "Only the label is revealed \u2014 your input was never exposed",
    icon: CircleCheck,
    privacyIcon: ShieldCheck,
  },
];

function getStageIndex(stage: InferenceStage): number {
  return STAGES.findIndex((s) => s.key === stage);
}

export function MPCProgressTracker({ stage, error, mpcStatus }: MPCProgressTrackerProps) {
  const currentIndex = getStageIndex(stage);

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-medium text-[var(--foreground)]">
          Inference Pipeline
        </h3>
        {stage !== "idle" && stage !== "error" && stage !== "complete" && (
          <span className="text-[10px] text-[var(--accent)] tabular-nums">
            Step {Math.min(currentIndex + 1, STAGES.length)} of {STAGES.length}
          </span>
        )}
        {stage === "complete" && (
          <span className="text-[10px] text-green-400">Complete</span>
        )}
      </div>

      <div className="space-y-1">
        {STAGES.map((s, i) => {
          const isActive = s.key === stage;
          const isComplete = currentIndex > i || stage === "complete";
          const isPending = !isActive && !isComplete;
          const Icon = s.icon;

          return (
            <div key={s.key} className="flex gap-3 relative">
              {/* Vertical connector */}
              {i < STAGES.length - 1 && (
                <div
                  className={`absolute left-[11px] top-[24px] w-px h-[calc(100%-12px)] ${
                    isComplete ? "bg-[var(--accent)]" : "bg-[var(--border)]"
                  }`}
                />
              )}

              {/* Icon */}
              <div className="flex-shrink-0 relative z-10 pt-0.5">
                {isComplete ? (
                  <div className="h-[22px] w-[22px] rounded-md bg-[var(--accent)] flex items-center justify-center">
                    <Icon className="h-3 w-3 text-white" strokeWidth={2.5} />
                  </div>
                ) : isActive ? (
                  <div className="h-[22px] w-[22px] rounded-md border-2 border-[var(--accent)] bg-[var(--accent)]/10 flex items-center justify-center pulse-ring">
                    <Icon className="h-3 w-3 text-[var(--accent)]" strokeWidth={2.5} />
                  </div>
                ) : (
                  <div className="h-[22px] w-[22px] rounded-md border border-[var(--border)] bg-[var(--muted)] flex items-center justify-center">
                    <Icon className="h-3 w-3 text-[var(--muted-foreground)]" strokeWidth={2} />
                  </div>
                )}
              </div>

              {/* Content */}
              <div className={`pb-3 min-w-0 flex-1 ${isPending ? "opacity-60" : ""}`}>
                <p
                  className={`text-xs font-medium leading-none ${
                    isActive
                      ? "text-[var(--foreground)]"
                      : isComplete
                        ? "text-[var(--accent)]"
                        : "text-[var(--muted-foreground)]"
                  }`}
                >
                  {s.label}
                </p>

                {/* Description — always visible */}
                <p className="text-[11px] text-[var(--muted-foreground)] mt-1 leading-relaxed">
                  {s.description}
                </p>

                {/* Technical detail */}
                <p className="text-[11px] text-[var(--muted-foreground)] mt-0.5 font-mono leading-relaxed opacity-70">
                  {s.technical}
                </p>

                {/* Privacy note */}
                <div className="flex items-center gap-1 mt-1">
                  <s.privacyIcon className="h-2.5 w-2.5 text-[var(--accent)] opacity-70 flex-shrink-0" strokeWidth={2} />
                  <p className="text-[11px] text-[var(--accent)] opacity-70 leading-none">
                    {s.privacy}
                  </p>
                </div>

                {/* MPC sub-status when active */}
                {isActive && s.key === "mpc" && mpcStatus && (
                  <div className="mt-1.5 rounded-md bg-[var(--accent)]/5 border border-[var(--accent)]/10 px-2 py-1">
                    <p className="text-[10px] text-[var(--accent)]">
                      {MPC_STATUS_LABELS[mpcStatus]}
                    </p>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {error && (
        <div className="mt-3 rounded-lg bg-red-500/10 border border-red-500/20 p-2.5">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
}
