import { NextRequest, NextResponse } from "next/server";

const stripTrailingSlash = (value: string) => value.replace(/\/+$/, "");

const getBackendBase = () => {
  const explicit = process.env.API_PROXY_TARGET?.trim();
  if (explicit && explicit.trim()) {
    return stripTrailingSlash(explicit.trim());
  }
  const nextPublic = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (nextPublic && /^https?:\/\//i.test(nextPublic)) {
    return stripTrailingSlash(nextPublic);
  }
  return "http://api:8000";
};

const getRequestTimeoutMs = () => {
  const fromEnv = process.env.API_PROXY_TIMEOUT_MS?.trim();
  const parsed = fromEnv ? Number(fromEnv) : NaN;
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }
  return 10000;
};

const makeProxyResponse = async (response: Response) => {
  const headers = sanitizeResponseHeaders(response.headers);
  const body = response.body ? response.body : null;
  return new NextResponse(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
};

const sanitizeResponseHeaders = (headers: Headers) => {
  const nextHeaders = new Headers();
  headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    if (lower === "connection" || lower === "keep-alive" || lower === "transfer-encoding") {
      return;
    }
    nextHeaders.set(key, value);
  });
  return nextHeaders;
};

const normalizePathSegments = (path: string[] | string | undefined | null) => {
  if (!path) return [];
  if (Array.isArray(path)) return path.filter(Boolean);
  return path.split("/").filter(Boolean);
};

const forwardRequest = async (req: NextRequest, path: string[] | string | undefined | null) => {
  const method = req.method.toUpperCase();
  const targetBase = getBackendBase();
  const normalizedPath = normalizePathSegments(path);
  const target = new URL(`${targetBase}/${normalizedPath.join("/")}`);
  target.search = req.nextUrl.search;

  const outgoingHeaders = new Headers(req.headers);
  outgoingHeaders.delete("host");
  outgoingHeaders.delete("connection");

  const hasBody = method !== "GET" && method !== "HEAD" && req.body !== null;
  const init: RequestInit = {
    method,
    headers: outgoingHeaders,
    redirect: "manual",
  };

  if (hasBody) {
    init.body = req.body;
    (init as RequestInit & { duplex?: "half" }).duplex = "half";
  }

  const controller = new AbortController();
  const timeoutMs = getRequestTimeoutMs();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(target.toString(), { ...init, signal: controller.signal });
    return makeProxyResponse(response);
  } finally {
    clearTimeout(timeout);
  }
};

export const runtime = "nodejs";

export async function GET(req: NextRequest, context: { params: Promise<{ path?: string[] | string } > }) {
  const { path } = await context.params;
  return forwardRequest(req, path);
}

export async function POST(req: NextRequest, context: { params: Promise<{ path?: string[] | string } > }) {
  const { path } = await context.params;
  return forwardRequest(req, path);
}

export async function PUT(req: NextRequest, context: { params: Promise<{ path?: string[] | string } > }) {
  const { path } = await context.params;
  return forwardRequest(req, path);
}

export async function PATCH(req: NextRequest, context: { params: Promise<{ path?: string[] | string } > }) {
  const { path } = await context.params;
  return forwardRequest(req, path);
}

export async function DELETE(req: NextRequest, context: { params: Promise<{ path?: string[] | string } > }) {
  const { path } = await context.params;
  return forwardRequest(req, path);
}
