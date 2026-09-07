export type ConversionStatus = "completed" | "failed";

export interface OrientedBoundingBox {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  rotation: number;
}

export interface DetectedSymbol {
  id: string;
  label: string;
  confidence: number;
  boundingBox: OrientedBoundingBox;
}

export interface PdfPageResult {
  pageNumber: number;
  width: number;
  height: number;
  detectedSymbols: DetectedSymbol[];
}

export interface PatternInstruction {
  id: string;
  row: number;
  text: string;
  confidence: number | null;
}

export interface ConversionResponse {
  status: ConversionStatus;
  mock: boolean;
  title: string;
  instructions: PatternInstruction[];
  pages: PdfPageResult[];
  averageConfidence: number | null;
  message?: string;
}

const MOCK_INSTRUCTIONS: PatternInstruction[] = [
  { id: "row-1", row: 1, text: "Make a magic ring. Ch 3 (counts as dc), work 11 dc into ring. Join with sl st to top of ch-3. (12 sts)", confidence: null },
  { id: "row-2", row: 2, text: "Ch 3, dc in same st, 2 dc in each st around. Join with sl st. (24 sts)", confidence: null },
  { id: "row-3", row: 3, text: "Ch 3, 2 dc in next st, *dc in next st, 2 dc in next st; repeat from * around. Join with sl st. (36 sts)", confidence: null },
  { id: "row-4", row: 4, text: "Ch 1, sc in same st, ch 3, skip 2 sts, *sc in next st, ch 3, skip 2 sts; repeat from * around. Join with sl st. (12 loops)", confidence: null },
  { id: "row-5", row: 5, text: "Sl st into first ch-3 space. Ch 3, work 4 dc in same space, sc in next space, *5 dc in next space, sc in next space; repeat from * around. Join and fasten off.", confidence: null },
];

function isPdf(file: File) {
  return file.type === "application/pdf" || (file.type === "" && file.name.toLowerCase().endsWith(".pdf"));
}

export async function convertPattern(file: File): Promise<ConversionResponse> {
  if (!isPdf(file)) throw new Error("Please select a valid PDF file.");

  await new Promise((resolve) => setTimeout(resolve, 1400));

  return {
    status: "completed",
    mock: true,
    title: "Simple Scalloped Coaster",
    instructions: MOCK_INSTRUCTIONS.map((instruction) => ({ ...instruction })),
    pages: [],
    averageConfidence: null,
    message: "Demo pattern loaded. Your PDF has not been analyzed.",
  };
}
