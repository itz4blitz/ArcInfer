type DebugLevel = "debug" | "info" | "warn" | "error";

function enabled() {
  if (typeof process !== "undefined") {
    const v = process.env.NEXT_PUBLIC_DEBUG;
    if (v === "1" || v === "true") return true;
  }
  if (typeof window !== "undefined") {
    const v = window.localStorage.getItem("arcinfer.debug");
    if (v === "1" || v === "true") return true;
  }
  return false;
}

/**
 * Structured console logging gated behind NEXT_PUBLIC_DEBUG=1
 * or localStorage arcinfer.debug=1. No-op otherwise.
 */
export function debugEvent(
  scope: string,
  message: string,
  data?: unknown,
  level: DebugLevel = "debug"
) {
  if (!enabled()) return;

  const prefix = `[arcinfer:${scope}] ${message}`;
  if (level === "error") console.error(prefix, data);
  else if (level === "warn") console.warn(prefix, data);
  else if (level === "info") console.info(prefix, data);
  else console.debug(prefix, data);
}
