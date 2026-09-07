"use client";

import { useRef, useState, type ChangeEvent, type DragEvent, type KeyboardEvent } from "react";

interface PdfUploadProps {
  file: File | null;
  error: string | null;
  disabled?: boolean;
  onFileChange: (file: File | null, error?: string) => void;
}

const PDF_ERROR = "That file isn’t a PDF. Please choose a file ending in .pdf.";

function isPdf(file: File) {
  return file.type === "application/pdf" || (file.type === "" && file.name.toLowerCase().endsWith(".pdf"));
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function PdfUpload({ file, error, disabled = false, onFileChange }: PdfUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const validateAndSelect = (selectedFile?: File) => {
    if (!selectedFile) return;
    if (!isPdf(selectedFile)) {
      onFileChange(null, PDF_ERROR);
      return;
    }
    onFileChange(selectedFile);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    validateAndSelect(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (!disabled) validateAndSelect(event.dataTransfer.files[0]);
  };

  const openFilePicker = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  };

  return (
    <div>
      <input ref={inputRef} id="pdf-upload" type="file" accept="application/pdf,.pdf" className="sr-only" onChange={handleInputChange} disabled={disabled} aria-label="Choose a crochet chart PDF" aria-describedby={error ? "upload-error" : undefined} />

      {file ? (
        <div className="rounded-[1.4rem] border-2 border-black bg-white p-4 sm:p-5">
          <div className="flex items-center gap-4">
            <div className="grid h-14 w-12 shrink-0 place-items-center rounded-lg border-2 border-black bg-[#f1f0eb]" aria-hidden="true"><span className="text-[10px] font-black tracking-wider">PDF</span></div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-bold">{file.name}</p>
              <p className="mt-1 text-xs text-black/50">{formatFileSize(file.size)} · Ready to convert</p>
            </div>
            <button type="button" onClick={() => onFileChange(null)} disabled={disabled} className="grid h-10 w-10 shrink-0 place-items-center rounded-full border border-black/20 transition hover:bg-black hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40" aria-label={`Remove ${file.name}`}><CloseIcon /></button>
          </div>
          <button type="button" onClick={openFilePicker} disabled={disabled} className="mt-4 w-full rounded-xl border border-dashed border-black/30 py-2.5 text-xs font-bold transition hover:border-black hover:bg-black/3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40">Choose a different PDF</button>
        </div>
      ) : (
        <div role="button" tabIndex={disabled ? -1 : 0} onClick={openFilePicker} onKeyDown={handleKeyDown} onDragEnter={(event) => { event.preventDefault(); if (!disabled) setIsDragging(true); }} onDragOver={(event) => event.preventDefault()} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setIsDragging(false); }} onDrop={handleDrop} aria-label="Upload crochet chart PDF" aria-disabled={disabled} className={`group cursor-pointer rounded-[1.4rem] border-2 border-dashed px-5 py-9 text-center transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-black focus-visible:ring-offset-4 ${isDragging ? "scale-[1.01] border-black bg-[#eceae2]" : "border-black/25 bg-white hover:border-black/60 hover:bg-[#fbfaf7]"} ${disabled ? "cursor-not-allowed opacity-50" : ""}`}>
          <span className="mx-auto grid h-14 w-14 place-items-center rounded-full border-2 border-black bg-[#f1f0eb] transition-transform group-hover:-translate-y-1"><UploadIcon /></span>
          <p className="font-hand mt-4 text-lg font-bold">Drop your chart here</p>
          <p id="upload-help" className="mt-1.5 text-sm text-black/50">or click to browse · PDF only</p>
        </div>
      )}

      <div aria-live="polite" aria-atomic="true">
        {error && <p id="upload-error" role="alert" className="mt-3 flex items-start gap-2 text-sm font-bold text-red-700"><AlertIcon />{error}</p>}
      </div>
    </div>
  );
}

function UploadIcon() {
  return <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" aria-hidden="true"><path d="M12 16V4m0 0L7 9m5-5 5 5M5 15v4h14v-4" /></svg>;
}

function CloseIcon() {
  return <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="2" aria-hidden="true"><path d="m5 5 10 10M15 5 5 15" /></svg>;
}

function AlertIcon() {
  return <svg viewBox="0 0 20 20" className="mt-0.5 h-4 w-4 shrink-0" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" aria-hidden="true"><circle cx="10" cy="10" r="8" /><path d="M10 6v5m0 3h.01" /></svg>;
}
