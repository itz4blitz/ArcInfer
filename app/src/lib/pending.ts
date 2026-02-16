export type PendingNetwork = "devnet" | "localnet";

export type PendingComputation = {
  network: PendingNetwork;
  programId: string;
  clusterOffset: number;
  submitTx: string;
  computationOffset: string;
  computationAccount: string;
  submittedAt: number;
  textLen: number;
};

const KEY = "arcinfer.pending.v1";

export function loadPendingComputation(): PendingComputation | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object") return null;
    const p = parsed as Record<string, unknown>;
    const network = p.network;
    if (network !== "devnet" && network !== "localnet") return null;
    if (typeof p.programId !== "string") return null;
    if (typeof p.clusterOffset !== "number") return null;
    if (typeof p.submitTx !== "string") return null;
    if (typeof p.computationOffset !== "string") return null;
    if (typeof p.computationAccount !== "string") return null;
    if (typeof p.submittedAt !== "number") return null;
    if (typeof p.textLen !== "number") return null;

    return {
      network,
      programId: p.programId,
      clusterOffset: p.clusterOffset,
      submitTx: p.submitTx,
      computationOffset: p.computationOffset,
      computationAccount: p.computationAccount,
      submittedAt: p.submittedAt,
      textLen: p.textLen,
    };
  } catch {
    return null;
  }
}

export function savePendingComputation(p: PendingComputation) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY, JSON.stringify(p));
}

export function clearPendingComputation() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(KEY);
}
