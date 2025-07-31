import { QueryClient, QueryFunction } from "@tanstack/react-query";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  // Ensure trailing slash for API endpoints (FastAPI behavior)
  let finalUrl = url;
  if (url.startsWith('/api/') && !url.endsWith('/') && !url.includes('?')) {
    // Don't add trailing slash to URLs with IDs like /api/projects/123
    const parts = url.split('/');
    const lastPart = parts[parts.length - 1];
    // If the last part doesn't look like an ID (no dashes, not 'health'), add trailing slash
    // Exception: Don't add trailing slash for DELETE requests with IDs
    if (!lastPart.includes('-') && lastPart !== 'health' && !(method === 'DELETE' && parts.length > 3)) {
      finalUrl = url + '/';
    }
  }
  
  const res = await fetch(finalUrl, {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  await throwIfResNotOk(res);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    let url = queryKey.join("/") as string;
    
    // Ensure trailing slash for API endpoints (FastAPI behavior)
    if (url.startsWith('/api/') && !url.endsWith('/') && !url.includes('?')) {
      const parts = url.split('/');
      const lastPart = parts[parts.length - 1];
      // If the last part doesn't look like an ID (no dashes, not 'health'), add trailing slash
      if (!lastPart.includes('-') && lastPart !== 'health') {
        url = url + '/';
      }
    }
    
    const res = await fetch(url, {
      credentials: "include",
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});
