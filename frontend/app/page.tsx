import Image from "next/image";
import { PatternConverter } from "./components/pattern-converter";

export default function Home() {
  return (
    <main className="relative min-h-svh overflow-hidden bg-[#f7f6f2] text-[#171717]">
      <div className="pointer-events-none absolute inset-0 craft-grid opacity-55" />
      <div className="pointer-events-none absolute -left-28 top-64 h-72 w-72 rounded-full border border-black/8" />
      <div className="pointer-events-none absolute -right-32 top-24 h-96 w-96 rounded-full border border-black/8" />

      <div className="relative mx-auto flex min-h-svh w-full max-w-6xl flex-col px-5 py-4 sm:px-8 lg:h-svh lg:px-12 lg:py-4">
        <header className="flex shrink-0 items-center justify-between border-b border-black/12 pb-3">
          <a href="#top" className="group flex items-center gap-3 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-4 focus-visible:ring-offset-[#f7f6f2]" aria-label="StitchScript home">
            <Image src="/cat-logo.svg" alt="" width={74} height={49} priority className="h-9 w-auto transition-transform group-hover:-rotate-2" />
            <span className="font-hand text-xl font-bold tracking-tight sm:text-2xl">stitched</span>
          </a>
          <span className="hidden items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-black/55 sm:flex">
            <span className="h-2 w-2 rounded-full bg-black" /> Demo workspace
          </span>
        </header>

        <section id="top" className="grid flex-1 items-center gap-8 py-9 lg:min-h-0 lg:grid-cols-[0.88fr_1.12fr] lg:gap-12 lg:py-6">
          <div className="max-w-xl">
            <p className="mb-3 flex items-center gap-3 text-xs font-bold uppercase tracking-[0.2em] text-black/55"><span className="block h-px w-8 bg-black/40" />Chart to written pattern</p>
            <h1 className="font-hand text-[clamp(3rem,6.5vw,5.5rem)] font-bold leading-[0.88] tracking-[-0.055em]">
              Make every
              <span className="relative mt-2 block w-fit">stitch clear.
                <svg className="absolute -bottom-4 left-1 h-4 w-[98%]" viewBox="0 0 400 20" aria-hidden="true" preserveAspectRatio="none"><path d="M3 13c96-8 233-9 394-5" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="5" /></svg>
              </span>
            </h1>
            <p className="mt-7 max-w-lg text-base leading-7 text-black/65 sm:text-lg sm:leading-7">Upload a crochet chart PDF and preview how stitched will turn visual symbols into an editable, text pattern.</p>
            <div className="mt-5 flex flex-wrap gap-x-7 gap-y-2 text-sm font-bold">
              <span className="flex items-center gap-2"><CheckMark /> PDF upload</span>
              <span className="flex items-center gap-2"><CheckMark /> Editable output</span>
              <span className="flex items-center gap-2"><CheckMark /> No account needed</span>
            </div>
          </div>
          <PatternConverter />
        </section>

        <footer className="flex shrink-0 flex-col gap-1 border-t border-black/12 py-3 text-xs text-black/50 sm:flex-row sm:items-center sm:justify-between">
          <p>Made for crocheters, one stitch at a time.</p><p>Prototype · No files are uploaded or stored</p>
        </footer>
      </div>
    </main>
  );
}

function CheckMark() {
  return <svg viewBox="0 0 20 20" className="h-4 w-4" aria-hidden="true"><path d="m4 10 4 4 8-9" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" /></svg>;
}
