import { PublicKey } from "@solana/web3.js";
import { PROGRAM_ID as DEFAULT_PROGRAM_ID } from "./constants";

export type NetworkKey = "devnet" | "localnet";

export interface ArcInferNetworkConfig {
  network: NetworkKey;
  rpcEndpoint: string;
  // Optional explicit PubSub websocket endpoint.
  // Useful for local validators where WS runs on a different port (typically rpcPort + 1).
  wsEndpoint?: string;
  clusterOffset: number;
  programId: PublicKey;
  walletMode: "phantom" | "burner";
  getExplorerTxUrl?: (sig: string) => string;
}

const DEVNET_RPC_DEFAULT = "https://api.devnet.solana.com";
const LOCALNET_RPC_DEFAULT = "http://127.0.0.1:8899";

function deriveWsEndpoint(rpcEndpoint: string): string | undefined {
  try {
    const url = new URL(rpcEndpoint);
    const wsProtocol = url.protocol === "https:" ? "wss:" : "ws:";

    // solana-test-validator defaults to `rpcPort + 1` for PubSub.
    // Wallet adapter's ConnectionProvider does not apply this heuristic,
    // so localnet subscriptions (onLogs/onSignature/etc) can silently never fire.
    const isLocalHost =
      url.hostname === "127.0.0.1" ||
      url.hostname === "localhost" ||
      url.hostname === "0.0.0.0";

    let port = url.port;
    if (isLocalHost && port) {
      const n = Number(port);
      if (Number.isFinite(n)) port = String(n + 1);
    }

    const wsUrl = new URL(url.toString());
    wsUrl.protocol = wsProtocol;
    if (port) wsUrl.port = port;
    return wsUrl.toString();
  } catch {
    return undefined;
  }
}

function parseClusterOffset(value: string | undefined, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseProgramId(value: string | undefined, fallback: PublicKey): PublicKey {
  if (!value) return fallback;
  return new PublicKey(value);
}

export function getNetworkConfig(network: NetworkKey): ArcInferNetworkConfig {
  if (network === "localnet") {
    const rpcEndpoint =
      process.env.NEXT_PUBLIC_LOCAL_RPC_URL || LOCALNET_RPC_DEFAULT;

    const wsEndpoint =
      process.env.NEXT_PUBLIC_LOCAL_WS_URL || deriveWsEndpoint(rpcEndpoint);
    const clusterOffset = parseClusterOffset(
      process.env.NEXT_PUBLIC_LOCAL_CLUSTER_OFFSET,
      // Local Arcium clusters created by `arcium test` use a small offset (0 by default).
      0
    );

    const programId = parseProgramId(
      process.env.NEXT_PUBLIC_LOCAL_PROGRAM_ID,
      DEFAULT_PROGRAM_ID
    );

    return {
      network,
      rpcEndpoint,
      wsEndpoint,
      clusterOffset,
      programId,
      walletMode: "burner",
    };
  }

  const rpcEndpoint = process.env.NEXT_PUBLIC_RPC_URL || DEVNET_RPC_DEFAULT;
  const clusterOffset = parseClusterOffset(
    process.env.NEXT_PUBLIC_CLUSTER_OFFSET,
    456
  );
  const programId = parseProgramId(process.env.NEXT_PUBLIC_PROGRAM_ID, DEFAULT_PROGRAM_ID);

  return {
    network,
    rpcEndpoint,
    clusterOffset,
    programId,
    walletMode: "phantom",
    getExplorerTxUrl: (sig) =>
      `https://explorer.solana.com/tx/${sig}?cluster=devnet`,
  };
}

const STORAGE_KEY = "arcinfer.network";

export function loadStoredNetwork(defaultNetwork: NetworkKey = "devnet"): NetworkKey {
  if (typeof window === "undefined") return defaultNetwork;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  return raw === "localnet" || raw === "devnet" ? raw : defaultNetwork;
}

export function storeNetwork(network: NetworkKey) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, network);
}
