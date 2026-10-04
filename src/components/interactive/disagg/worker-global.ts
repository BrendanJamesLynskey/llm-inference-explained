/**
 * Next's client build replaces `typeof window` with "object", so the vendored
 * engine's `typeof window !== 'undefined' ? window : globalThis` compiles to
 * plain `window`, which a Web Worker does not have. Give the worker one,
 * before the engine module runs (imported first by sim.worker.ts).
 */
(self as unknown as { window: typeof self }).window = self;

export {};
