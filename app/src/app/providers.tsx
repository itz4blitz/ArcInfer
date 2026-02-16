"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ConnectionProvider,
  WalletProvider,
} from "@solana/wallet-adapter-react";
import { WalletModalProvider } from "@solana/wallet-adapter-react-ui";
import {
  PhantomWalletAdapter,
  UnsafeBurnerWalletAdapter,
} from "@solana/wallet-adapter-wallets";

import {
  getNetworkConfig,
  loadStoredNetwork,
  storeNetwork,
  type ArcInferNetworkConfig,
  type NetworkKey,
} from "@/lib/networks";

import "@solana/wallet-adapter-react-ui/styles.css";

type NetworkContextValue = {
  network: NetworkKey;
  setNetwork: (n: NetworkKey) => void;
  config: ArcInferNetworkConfig;
};

const NetworkContext = createContext<NetworkContextValue | null>(null);

export function useArcInferNetwork(): NetworkContextValue {
  const ctx = useContext(NetworkContext);
  if (!ctx) {
    throw new Error("useArcInferNetwork must be used within Providers");
  }
  return ctx;
}

export function Providers({ children }: { children: ReactNode }) {
  const [network, setNetworkState] = useState<NetworkKey>("devnet");

  useEffect(() => {
    // Hydration-safe: load the persisted preference client-side.
    setNetworkState(loadStoredNetwork("devnet"));
  }, []);

  const setNetwork = (n: NetworkKey) => {
    setNetworkState(n);
    storeNetwork(n);
  };

  const config = useMemo(() => getNetworkConfig(network), [network]);

  const wallets = useMemo(() => {
    return config.walletMode === "burner"
      ? [new UnsafeBurnerWalletAdapter()]
      : [new PhantomWalletAdapter()];
  }, [config.walletMode]);

  return (
    <NetworkContext.Provider value={{ network, setNetwork, config }}>
      <ConnectionProvider
        endpoint={config.rpcEndpoint}
        config={{ commitment: "confirmed", wsEndpoint: config.wsEndpoint }}
        key={config.rpcEndpoint}
      >
        <WalletProvider wallets={wallets} autoConnect>
          <WalletModalProvider>{children}</WalletModalProvider>
        </WalletProvider>
      </ConnectionProvider>
    </NetworkContext.Provider>
  );
}
