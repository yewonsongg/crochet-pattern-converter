"use client";

import { useState } from "react";
import { convertPattern, type ConversionResponse } from "@/lib/mock-conversion";
import { ConversionResults } from "./conversion-results";
import { PdfUpload } from "./pdf-upload";

export function PatternConverter() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ConversionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (nextFile: File | null, validationError?: string) => {
    setFile(nextFile);
    setResult(null);
    setError(validationError ?? null);
  };

  const handleConvert = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    try {
      const response = await convertPattern(file);
      if (response.status !== "completed") throw new Error(response.message || "Conversion could not be completed.");
      setResult(response);
    } catch (conversionError) {
      setError(conversionError instanceof Error ? conversionError.message : "Something went wrong. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const startOver = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  if (result) return <ConversionResults result={result} onStartOver={startOver} />;

  return (
    <section aria-labelledby="converter-title" className="relative rounded-[2rem] border-2 border-black bg-[#efeee9] p-5 shadow-[7px_7px_0_#171717] sm:p-6">
      <span className="font-hand absolute -right-3 -top-3 grid h-11 w-11 rotate-6 place-items-center rounded-full border-2 border-black bg-white text-lg font-bold" aria-hidden="true">✦</span>
      <div className="mb-4">
        <p className="text-[10px] font-black uppercase tracking-[0.2em] text-black/45">Step 01</p>
        <h2 id="converter-title" className="font-hand mt-1 text-2xl font-bold sm:text-3xl">Upload your crochet chart</h2>
        <p className="mt-1 text-sm leading-5 text-black/55">Choose a PDF chart to load a sample written pattern.</p>
      </div>

      <PdfUpload file={file} error={error} disabled={isLoading} onFileChange={handleFileChange} />

      <button type="button" onClick={handleConvert} disabled={!file || isLoading} className="mt-4 flex min-h-12 w-full items-center justify-center gap-3 rounded-full bg-black px-6 text-sm font-black text-white transition hover:-translate-y-0.5 hover:shadow-[0_5px_0_rgb(0_0_0_/_0.2)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-4 disabled:cursor-not-allowed disabled:bg-black/20 disabled:text-black/45 disabled:shadow-none disabled:hover:translate-y-0">
        {isLoading ? <><Spinner /> Preparing demo pattern…</> : <>Convert to written pattern <ArrowIcon /></>}
      </button>

      {isLoading && <p role="status" aria-live="polite" className="mt-3 text-center text-xs font-bold text-black/55">This is a simulated conversion. No model is analyzing your PDF.</p>}
      <p className="mt-5 text-center text-[11px] leading-5 text-black/45"><span aria-hidden="true">♢</span> Prototype mode — your file stays in this browser</p>
    </section>
  );
}

function Spinner() {
  return <svg viewBox="0 0 24 24" className="h-5 w-5 animate-spin" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity=".25" strokeWidth="3" /><path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeLinecap="round" strokeWidth="3" /></svg>;
}

function ArrowIcon() {
  return <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" aria-hidden="true"><path d="M3 10h14m-5-5 5 5-5 5" /></svg>;
}
