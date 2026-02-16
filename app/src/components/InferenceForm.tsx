"use client";

import { useState } from "react";
import { Sparkles } from "lucide-react";

interface InferenceFormProps {
  onSubmit: (text: string) => void;
  disabled: boolean;
  walletConnected: boolean;
  modelReady: boolean;
}

export function InferenceForm({
  onSubmit,
  disabled,
  walletConnected,
  modelReady,
}: InferenceFormProps) {
  const [text, setText] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !disabled) {
      onSubmit(text.trim());
    }
  };

  const placeholder = "This movie was absolutely fantastic!";

  let buttonText = "Classify Sentiment";
  if (!walletConnected) buttonText = "Connect Wallet First";
  else if (!modelReady) buttonText = "Loading Model...";
  else if (disabled) buttonText = "Processing...";

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label
          htmlFor="text-input"
          className="block text-sm font-medium text-[var(--foreground)] mb-2"
        >
          Enter text for private sentiment analysis
        </label>
        <textarea
          id="text-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={placeholder}
          rows={3}
          className="w-full rounded-lg border border-[var(--border)] bg-[var(--muted)]
                     px-4 py-3 text-[var(--foreground)] placeholder:text-[var(--muted-foreground)]
                     focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-transparent
                     resize-none transition-all"
          disabled={disabled}
        />
        <p className="mt-1.5 text-xs text-[var(--muted-foreground)]">
          Your text is processed locally in your browser. Only the encrypted
          representation enters the network.
        </p>
      </div>

      <button
        type="submit"
        disabled={disabled || !text.trim() || !walletConnected || !modelReady}
        className="w-full rounded-lg bg-[var(--accent)] px-4 py-3 text-sm font-medium
                   text-white transition-all hover:bg-[var(--accent-hover)]
                   disabled:opacity-40 disabled:cursor-not-allowed
                   flex items-center justify-center gap-2"
      >
        <Sparkles className="h-4 w-4" />
        {buttonText}
      </button>

      {/* Quick examples */}
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-[var(--muted-foreground)]">Try:</span>
        {["I love this!", "Terrible experience.", "Pretty good overall."].map(
          (example) => (
            <button
              key={example}
              type="button"
              onClick={() => setText(example)}
              disabled={disabled}
              className="text-xs rounded-full border border-[var(--border)] px-3 py-1
                         text-[var(--muted-foreground)] hover:text-[var(--foreground)]
                         hover:border-[var(--accent)] transition-all
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {example}
            </button>
          )
        )}
      </div>
    </form>
  );
}
