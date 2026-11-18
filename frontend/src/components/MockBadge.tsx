import { ReactNode } from "react";
import { MOCK_LABEL } from "@/lib/episodes";

export function MockBadge({ size = "sm" }: { size?: "sm" | "md" }) {
  const scale = size === "md" ? "px-2.5 py-1 text-sm" : "px-1.5 py-0.5 text-[10px]";
  return (
    <span
      className={`rounded-md bg-amber-400 font-mono font-black tracking-widest text-amber-950 ${scale}`}
    >
      {MOCK_LABEL}
    </span>
  );
}

/** A full-width strip saying which part of a view is not a simulation result. */
export function MockNotice({ children }: { children: ReactNode }) {
  return (
    <div
      role="note"
      className="flex flex-wrap items-center gap-3 rounded-2xl border-2 border-amber-400/70 bg-amber-950/60 px-5 py-3"
    >
      <MockBadge size="md" />
      <span className="text-sm text-amber-100">{children}</span>
    </div>
  );
}
