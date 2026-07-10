import type { HealthResponse, StatisticsResponse, TokenizeResponse } from "../types";

// Configurable so the frontend can be deployed (e.g. to Netlify) pointed
// at a backend hosted anywhere -- see .env.example and README.md.
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      ...init,
    });
  } catch {
    throw new ApiError(
      `Could not reach the API at ${API_BASE_URL}. Is the backend running?`,
      0,
    );
  }

  if (!response.ok) {
    let detail = response.statusText || `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // response body wasn't JSON -- keep the status-text fallback above.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function getStatistics(): Promise<StatisticsResponse> {
  return request<StatisticsResponse>("/api/statistics");
}

export function tokenizeText(text: string): Promise<TokenizeResponse> {
  return request<TokenizeResponse>("/api/tokenize", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export type DownloadFile = "vocab.json" | "merges.json" | "tokenizer.json";

export function downloadUrl(filename: DownloadFile): string {
  return `${API_BASE_URL}/tokenizer/${filename}`;
}
