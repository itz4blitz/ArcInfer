"use client";

import { ThumbsUp, ThumbsDown, ExternalLink } from "lucide-react";

interface ResultDisplayProps {
  result: {
    sentiment: "positive" | "negative";
    class: number;
    text: string;
    tx?: string;
  } | null;
  getExplorerTxUrl?: (sig: string) => string;
}

export function ResultDisplay({ result, getExplorerTxUrl }: ResultDisplayProps) {
  if (!result) return null;

  const isPositive = result.sentiment === "positive";

  return (
    <div
      className={`rounded-xl border p-4 space-y-3 ${
        isPositive
          ? "border-green-500/20 bg-green-500/5"
          : "border-red-500/20 bg-red-500/5"
      }`}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-[var(--foreground)]">
          Classification Result
        </h3>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium ${
            isPositive
              ? "bg-green-500/10 text-green-400 border border-green-500/20"
              : "bg-red-500/10 text-red-400 border border-red-500/20"
          }`}
        >
          {isPositive ? (
            <ThumbsUp className="h-2.5 w-2.5" />
          ) : (
            <ThumbsDown className="h-2.5 w-2.5" />
          )}
          {result.sentiment.toUpperCase()}
        </span>
      </div>

      <blockquote className="border-l-2 border-[var(--accent)]/30 pl-3 text-xs text-[var(--muted-foreground)] italic">
        &ldquo;{result.text}&rdquo;
      </blockquote>

      {result.tx && getExplorerTxUrl && (
        <div className="pt-2 border-t border-[var(--border)]">
          <a
            href={getExplorerTxUrl(result.tx)}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-[11px] text-[var(--accent)] hover:underline"
          >
            <ExternalLink className="h-3 w-3" />
            View transaction on Solana Explorer
          </a>
        </div>
      )}
    </div>
  );
}
