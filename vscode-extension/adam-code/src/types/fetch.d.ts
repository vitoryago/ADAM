/**
 * Minimal type declarations for the global fetch API.
 *
 * Node.js 18+ (used by VS Code) ships native fetch, but @types/node v16
 * doesn't include the types. This declaration bridges the gap so TypeScript
 * can compile without adding a runtime polyfill.
 */

interface FetchResponse {
    ok: boolean;
    status: number;
    statusText: string;
    headers: Headers;
    body: ReadableStream<Uint8Array> | null;
    text(): Promise<string>;
    json(): Promise<any>;
}

interface ReadableStream<R = any> {
    readonly locked: boolean;
    cancel(reason?: any): Promise<void>;
    getReader(): ReadableStreamDefaultReader<R>;
}

interface ReadableStreamDefaultReader<R = any> {
    readonly closed: Promise<undefined>;
    cancel(reason?: any): Promise<void>;
    read(): Promise<ReadableStreamDefaultReadResult<R>>;
    releaseLock(): void;
}

type ReadableStreamDefaultReadResult<T> =
    | { done: false; value: T }
    | { done: true; value: undefined };

declare function fetch(input: string, init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
    signal?: AbortSignal;
}): Promise<FetchResponse>;
