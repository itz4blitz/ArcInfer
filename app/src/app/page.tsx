"use client";

import { useCallback, useEffect, useState } from "react";
import { useAnchorWallet, useConnection, useWallet } from "@solana/wallet-adapter-react";
import { WalletMultiButton } from "@solana/wallet-adapter-react-ui";
import * as anchor from "@coral-xyz/anchor";
import {
  Shield,
  Brain,
  Lock,
  Server,
  Github,
} from "lucide-react";

import { InferenceForm } from "@/components/InferenceForm";
import {
  MPCProgressTracker,
  type InferenceStage,
} from "@/components/MPCProgressTracker";
import { ResultDisplay } from "@/components/ResultDisplay";
import {
  initEmbedding,
  embed,
  resetEmbedding,
  type EmbeddingProgress,
} from "@/lib/embedding";
import { loadPcaParams, pcaTransform } from "@/lib/pca";
import { createEncryptionContext, encryptFeatures } from "@/lib/arcium";
import {
  getProgram,
  submitClassification,
  awaitResult,
  fetchMXEPublicKey,
  type MpcStatus,
} from "@/lib/program";

import {
  loadPendingComputation,
  savePendingComputation,
  clearPendingComputation,
  type PendingComputation,
} from "@/lib/pending";

import { useArcInferNetwork } from "@/app/providers";


interface ClassificationResult {
  sentiment: "positive" | "negative";
  class: number;
  text: string;
  tx?: string;
}

export default function Home() {
  const { connection } = useConnection();
  const wallet = useWallet();
  const anchorWallet = useAnchorWallet();
  const { network, setNetwork, config } = useArcInferNetwork();

  const [mounted, setMounted] = useState(false);
  const [modelProgress, setModelProgress] = useState<EmbeddingProgress>({
    status: "loading",
    progress: 0,
  });

  useEffect(() => setMounted(true), []);
  const [stage, setStage] = useState<InferenceStage>("idle");
  const [error, setError] = useState<string>();
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [crossOriginOk, setCrossOriginOk] = useState<boolean | null>(null);

  const [pending, setPending] = useState<PendingComputation | null>(null);
  const [mpcStatus, setMpcStatus] = useState<MpcStatus | undefined>();

  // Localnet convenience: ensure the burner wallet has funds.
  const [airdropDoneFor, setAirdropDoneFor] = useState<string | null>(null);

  useEffect(() => {
    if (network !== "localnet") return;
    const publicKey = wallet.publicKey;
    if (!publicKey) return;
    const pk = publicKey.toBase58();
    if (airdropDoneFor === pk) return;

    let cancelled = false;
    (async () => {
      try {
        const bal = await connection.getBalance(publicKey, "confirmed");
        if (bal < 0.5 * anchor.web3.LAMPORTS_PER_SOL) {
          const sig = await connection.requestAirdrop(
            publicKey,
            2 * anchor.web3.LAMPORTS_PER_SOL
          );
          await connection.confirmTransaction(sig, "confirmed");
        }
        if (!cancelled) setAirdropDoneFor(pk);
      } catch {
        // Ignore: local validator may not be running yet.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [network, wallet.publicKey, connection, airdropDoneFor]);

  useEffect(() => {
    initEmbedding(setModelProgress).catch((err) => {
      console.error("Failed to load embedding model:", err);
    });
  }, []);


  useEffect(() => {
    // Used only for user-facing diagnostics.
    if (typeof window === "undefined") return;
    setCrossOriginOk(window.crossOriginIsolated === true);
  }, []);

  useEffect(() => {
    setPending(loadPendingComputation());
  }, [network]);

  const clearPending = useCallback(() => {
    clearPendingComputation();
    setPending(null);
  }, []);

  const resumePending = useCallback(async () => {
    const p = loadPendingComputation();
    if (!p) return;
    if (p.network !== network) return;
    if (p.programId !== config.programId.toBase58()) return;
    if (p.clusterOffset !== config.clusterOffset) return;

    setStage("mpc");
    setError(undefined);
    setResult(null);
    setMpcStatus(undefined);

    try {
      const computationOffset = new anchor.BN(p.computationOffset);
      const { classValue } = await awaitResult(
        connection,
        computationOffset,
        { programId: config.programId, clusterOffset: config.clusterOffset },
        {
          timeoutMs: network === "localnet" ? 30_000 : 90_000,
          onStatus: (status) => setMpcStatus(status),
        }
      );

      setStage("complete");
      setResult({
        sentiment: classValue === 0 ? "positive" : "negative",
        class: classValue,
        text: "(previously submitted text not stored)",
        tx: p.submitTx,
      });
      clearPendingComputation();
      setPending(null);
    } catch (err) {
      setStage("error");
      setError(err instanceof Error ? err.message : String(err));
      setPending(loadPendingComputation());
    }
  }, [connection, config.clusterOffset, config.programId, network]);

  const retryModelLoad = useCallback(() => {
    resetEmbedding();
    initEmbedding(setModelProgress).catch((err) => {
      console.error("Failed to load embedding model:", err);
    });
  }, []);


  const handleClassify = useCallback(
    async (text: string) => {
      if (!anchorWallet?.publicKey || !anchorWallet.signTransaction) return;

      setStage("embedding");
      setError(undefined);
      setResult(null);
      setMpcStatus(undefined);

      try {
        const embedding = await embed(text);

        setStage("pca");
        const pcaParams = await loadPcaParams();
        const quantized = pcaTransform(embedding, pcaParams);

        setStage("encrypting");
        const provider = new anchor.AnchorProvider(connection, anchorWallet, {
          commitment: "confirmed",
        });
        const mxePublicKey = await fetchMXEPublicKey(provider, config.programId);
        const encCtx = await createEncryptionContext(mxePublicKey);
        const { ciphertexts, nonceValue } = encryptFeatures(
          quantized,
          encCtx.cipher
        );

        setStage("submitting");
        const program = getProgram(connection, anchorWallet);
        const { tx, computationOffset, computationAccount } =
          await submitClassification(
          program,
          anchorWallet,
          ciphertexts,
          encCtx.clientPublicKey,
          nonceValue,
          { programId: config.programId, clusterOffset: config.clusterOffset }
        );

        savePendingComputation({
          network,
          programId: config.programId.toBase58(),
          clusterOffset: config.clusterOffset,
          submitTx: tx,
          computationOffset: computationOffset.toString(),
          computationAccount: computationAccount.toBase58(),
          submittedAt: Date.now(),
          textLen: text.length,
        });
        setPending(loadPendingComputation());

        setStage("mpc");
        setMpcStatus(undefined);
        const { classValue } = await awaitResult(
          connection,
          computationOffset,
          { programId: config.programId, clusterOffset: config.clusterOffset },
          {
            timeoutMs: network === "localnet" ? 30_000 : 90_000,
            onStatus: (status) => setMpcStatus(status),
          }
        );

        setStage("complete");
        setResult({
          sentiment: classValue === 0 ? "positive" : "negative",
          class: classValue,
          text,
          tx,
        });

        clearPendingComputation();
        setPending(null);
      } catch (err) {
        console.error("Classification failed:", err);
        setStage("error");
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [anchorWallet, connection, config, network]
  );

  const isProcessing =
    stage !== "idle" && stage !== "complete" && stage !== "error";

  return (
    <div className="bg-glow min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] flex-shrink-0">
        <div className="flex items-center justify-between px-8 py-3">
          <div className="flex items-center gap-2.5">
            <div className="h-7 w-7 rounded-md bg-gradient-to-br from-[var(--accent)] to-[var(--accent-hover)] flex items-center justify-center">
              <Shield className="h-3.5 w-3.5 text-white" strokeWidth={2.5} />
            </div>
            <span className="text-sm font-semibold tracking-tight">
              ArcInfer
            </span>
            <span className="text-[10px] text-[var(--muted-foreground)] border-l border-[var(--border)] pl-2.5 ml-0.5 hidden sm:inline">
              Private AI on Arcium
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2">
              <span className="text-[10px] text-[var(--muted-foreground)]">
                Network
              </span>
              <select
                value={network}
                onChange={(e) => {
                  const v = e.target.value;
                  setNetwork(v === "localnet" ? "localnet" : "devnet");
                }}
                className="h-8 rounded-md border border-[var(--border)] bg-[var(--card)] px-2 text-[11px] text-[var(--foreground)]"
                aria-label="Select network"
              >
                <option value="devnet">Devnet</option>
                <option value="localnet">Localnet</option>
              </select>
            </div>
            {mounted && <WalletMultiButton />}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col px-8 py-6 gap-6">
        {/* Hero */}
        <div className="flex items-baseline justify-between">
          <div className="flex items-baseline gap-2">
            <h1 className="text-lg font-semibold tracking-tight">
              Encrypted AI Inference
            </h1>
            <p className="text-xs text-[var(--muted-foreground)]">
              Classify text without exposing it
            </p>
          </div>
          {network === "localnet" && (
            <p className="text-[10px] text-[var(--muted-foreground)]">
              Localnet &middot; <span className="font-mono">arcium test --detach</span>
            </p>
          )}
        </div>

        {/* What happens to your data */}
        <div className="grid grid-cols-4 gap-4">
          {[
            { icon: Brain, label: "Stays local", sub: "Text never leaves browser" },
            { icon: Lock, label: "Encrypted", sub: "Only ciphertext sent" },
            { icon: Server, label: "Split across nodes", sub: "No node sees full data" },
            { icon: Shield, label: "Only result returned", sub: "Input stays secret" },
          ].map((s, i) => {
            const Icon = s.icon;
            return (
              <div key={i} className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-3 flex items-center gap-3">
                <div className="h-8 w-8 rounded-md bg-[var(--accent)]/10 flex items-center justify-center flex-shrink-0">
                  <Icon className="h-4 w-4 text-[var(--accent)]" strokeWidth={2} />
                </div>
                <div>
                  <p className="text-[11px] font-medium text-[var(--foreground)]">{s.label}</p>
                  <p className="text-[10px] text-[var(--muted-foreground)] leading-tight">{s.sub}</p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Pending resume */}
        {pending && pending.network === network && (stage === "idle" || stage === "error") && (
          <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-4">
            <p className="text-xs text-[var(--muted-foreground)]">
              Pending MPC result found (submitted earlier). You can resume polling without resubmitting.
            </p>
            <div className="mt-3 flex items-center gap-2">
              <button
                type="button"
                onClick={resumePending}
                className="h-8 rounded-md bg-[var(--accent)] px-3 text-[11px] font-medium text-white hover:bg-[var(--accent-hover)]"
              >
                Resume
              </button>
              <button
                type="button"
                onClick={clearPending}
                className="h-8 rounded-md border border-[var(--border)] bg-[var(--card)] px-3 text-[11px] text-[var(--foreground)] hover:border-[var(--accent)]"
              >
                Clear
              </button>
              <span className="ml-auto text-[10px] text-[var(--muted-foreground)]">
                tx {pending.submitTx.slice(0, 6)}..{pending.submitTx.slice(-6)}
              </span>
            </div>
          </div>
        )}

        {/* Two-column: Input + Pipeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
          {/* Left: Input + status */}
          <div className="flex flex-col gap-4">
            <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-5">
              <InferenceForm
                onSubmit={handleClassify}
                disabled={isProcessing}
                walletConnected={!!wallet.publicKey}
                modelReady={modelProgress.status === "ready"}
              />
            </div>

            {/* Model loading */}
            {modelProgress.status === "loading" && (
              <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-3">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full border-2 border-[var(--accent)] border-t-transparent animate-spin" />
                  <p className="text-xs text-[var(--muted-foreground)]">
                    {modelProgress.message || "Loading AI model into browser..."}
                  </p>
                </div>
                {modelProgress.progress !== undefined && modelProgress.progress > 0 && (
                  <div className="mt-2 h-0.5 rounded-full bg-[var(--muted)] overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[var(--accent)] transition-all duration-300"
                      style={{ width: `${modelProgress.progress}%` }}
                    />
                  </div>
                )}
              </div>
            )}

            {crossOriginOk === false && (
              <div className="rounded-xl border border-yellow-500/20 bg-[var(--card)] p-3">
                <p className="text-xs text-yellow-400">
                  Browser isolation is off (COOP/COEP). The embedding model can hang loading.
                  Hard refresh the page (Cmd+Shift+R). If still broken, check that the response headers include COOP/COEP.
                </p>
              </div>
            )}

            {modelProgress.status === "error" && (
              <div className="rounded-xl border border-red-500/20 bg-[var(--card)] p-3 flex items-center justify-between gap-3">
                <p className="text-xs text-red-400">
                  {modelProgress.message || "Failed to load embedding model."}
                </p>
                <button
                  type="button"
                  onClick={retryModelLoad}
                  className="h-7 rounded-md border border-[var(--border)] bg-[var(--card)] px-2 text-[11px] text-[var(--foreground)] hover:border-[var(--accent)] flex-shrink-0"
                >
                  Retry
                </button>
              </div>
            )}


          </div>

          {/* Right: Pipeline + Result — always visible */}
          <div className="flex flex-col gap-4">
            <MPCProgressTracker stage={stage} error={error} mpcStatus={mpcStatus} />
            <ResultDisplay
              result={result}
              getExplorerTxUrl={config.getExplorerTxUrl}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-auto flex items-center justify-between text-[10px] text-[var(--muted-foreground)] pt-2 border-t border-[var(--border)]">
          <span>
            16{"\u2192"}16{"\u2192"}8{"\u2192"}2 NN &middot; 426 params &middot; Q16.16 fixed-point &middot; ~10 MPC rounds
          </span>
          <span className="flex items-center gap-2">
            Built on{" "}
            <a href="https://arcium.com" target="_blank" rel="noopener noreferrer" className="text-[var(--accent)] hover:underline">Arcium</a>
            {" "}&middot; Solana {network}
            <a
              href="https://github.com/itz4blitz/ArcInfer"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
            >
              <Github className="h-3.5 w-3.5" />
            </a>
          </span>
        </footer>
      </main>
    </div>
  );
}
