"use client";

import { useState } from "react";
import type { ConversionResponse, PatternInstruction } from "@/lib/mock-conversion";

interface ConversionResultsProps {
  result: ConversionResponse;
  onStartOver: () => void;
}

type CopyStatus = "idle" | "success" | "error";

export function ConversionResults({ result, onStartOver }: ConversionResultsProps) {
  const [instructions, setInstructions] = useState(result.instructions);
  const [copyStatus, setCopyStatus] = useState<CopyStatus>("idle");

  const updateInstruction = (id: string, text: string) => {
    setInstructions((current) => current.map((item) => item.id === id ? { ...item, text } : item));
    setCopyStatus("idle");
  };

  const copyPattern = async () => {
    const pattern = [result.title, "", ...instructions.map((instruction) => `Row ${instruction.row}: ${instruction.text}`)].join("\n");
    try {
      await navigator.clipboard.writeText(pattern);
      setCopyStatus("success");
    } catch {
      setCopyStatus("error");
    }
  };

  return (
    <section aria-labelledby="result-title" className="rounded-[2rem] border-2 border-black bg-white p-5 shadow-[7px_7px_0_#171717] sm:p-7">
      <div className="flex flex-col gap-4 border-b border-black/12 pb-5 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-black px-3 py-1 text-[10px] font-black uppercase tracking-[0.16em] text-white">Demo result</span>
            <span className="text-xs font-bold text-black/45">Mock conversion · PDF not analyzed</span>
          </div>
          <h2 id="result-title" className="font-hand text-2xl font-bold leading-tight sm:text-3xl">{result.title}</h2>
        </div>
        <button type="button" onClick={copyPattern} className="flex h-10 shrink-0 items-center justify-center gap-2 rounded-full border-2 border-black px-4 text-xs font-black transition hover:bg-black hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-2">
          <CopyIcon /> {copyStatus === "success" ? "Copied!" : "Copy pattern"}
        </button>
      </div>

      <p className="mt-5 text-sm leading-6 text-black/55">Edit any instruction below before copying your sample pattern.</p>
      <div className="mt-4 max-h-[22rem] space-y-3 overflow-y-auto pr-1">
        {instructions.map((instruction) => <InstructionEditor key={instruction.id} instruction={instruction} onChange={updateInstruction} />)}
      </div>

      <div className="mt-5 flex flex-col-reverse gap-3 border-t border-black/12 pt-5 sm:flex-row sm:items-center sm:justify-between">
        <p role="status" aria-live="polite" className={`min-h-5 text-xs font-bold ${copyStatus === "error" ? "text-red-700" : "text-black/55"}`}>
          {copyStatus === "success" && "Pattern copied to your clipboard."}
          {copyStatus === "error" && "Couldn’t access the clipboard. Please select and copy the text manually."}
        </p>
        <button type="button" onClick={onStartOver} className="rounded-full px-3 py-2 text-sm font-black underline decoration-2 underline-offset-4 transition hover:bg-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black">Start a new conversion</button>
      </div>
    </section>
  );
}

function InstructionEditor({ instruction, onChange }: { instruction: PatternInstruction; onChange: (id: string, text: string) => void }) {
  return (
    <div className="grid gap-2 rounded-xl border border-black/15 bg-[#f8f7f3] p-3 sm:grid-cols-[4.5rem_1fr] sm:gap-3">
      <label htmlFor={`instruction-${instruction.id}`} className="pt-2 text-xs font-black uppercase tracking-wider text-black/55">Row {instruction.row}</label>
      <textarea id={`instruction-${instruction.id}`} value={instruction.text} onChange={(event) => onChange(instruction.id, event.target.value)} rows={3} className="min-h-20 w-full resize-y rounded-lg border border-transparent bg-white px-3 py-2 text-sm leading-6 outline-none transition focus:border-black focus:ring-2 focus:ring-black/10" />
    </div>
  );
}

function CopyIcon() {
  return <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="6" y="6" width="10" height="10" rx="2" /><path d="M13 6V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h1" /></svg>;
}
