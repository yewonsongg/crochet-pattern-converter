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
  pageNumber: number;
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

interface ApiErrorBody {
  detail?: string;
}

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export async function convertPattern(file: File): Promise<ConversionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/convert`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Could not reach the conversion service. Make sure the FastAPI server is running and try again.");
  }

  if (!response.ok) {
    const body = await readErrorBody(response);
    throw new Error(body?.detail || `Conversion failed with status ${response.status}.`);
  }

  const result: unknown = await response.json();
  if (!isConversionResponse(result)) {
    throw new Error("The conversion service returned an unexpected response.");
  }

  return result;
}

async function readErrorBody(response: Response): Promise<ApiErrorBody | null> {
  try {
    return (await response.json()) as ApiErrorBody;
  } catch {
    return null;
  }
}

function isConversionResponse(value: unknown): value is ConversionResponse {
  if (!value || typeof value !== "object") return false;
  const response = value as Partial<ConversionResponse>;
  return (
    (response.status === "completed" || response.status === "failed") &&
    typeof response.mock === "boolean" &&
    typeof response.title === "string" &&
    Array.isArray(response.instructions) &&
    Array.isArray(response.pages) &&
    (typeof response.averageConfidence === "number" || response.averageConfidence === null)
  );
}
