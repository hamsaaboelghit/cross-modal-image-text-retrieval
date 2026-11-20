"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  ArrowDown, ArrowRight, ArrowUpRight, BadgeCheck, BrainCircuit,
  ChevronDown, CircleAlert, Clock3, Database, Image as ImageIcon,
  LoaderCircle, RefreshCw, Search, Sparkles, Upload, X,
} from "lucide-react";

type Mode = "text" | "image";
type ServiceStatus = "checking" | "ready" | "degraded";
type ModelOption = { id: string; name: string; loaded: boolean };
type SearchResult = {
  rank: number;
  similarity: number;
  image_id?: string;
  image_url?: string;
  caption?: string | null;
  result?: unknown;
};
type SearchResponse = {
  mode: string;
  model: string;
  count: number;
  elapsed_ms: number;
  results: SearchResult[];
};
type HealthPayload = { status?: string; startup_error?: string | null; model_loaded?: boolean; gallery_loaded?: boolean };

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export default function Home() {
  const [mode, setMode] = useState<Mode>("text");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [minSimilarity, setMinSimilarity] = useState(-1);
  const [model, setModel] = useState("baseline");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [status, setStatus] = useState<ServiceStatus>("checking");
  const [statusMessage, setStatusMessage] = useState("Checking model and retrieval gallery…");
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshStatus = useCallback(async () => {
    setStatus("checking");
    try {
      const [modelsResponse, healthResponse] = await Promise.all([
        fetch(`${API_BASE}/models`, { cache: "no-store" }),
        fetch(`${API_BASE}/health`, { cache: "no-store" }),
      ]);
      if (!modelsResponse.ok || !healthResponse.ok) throw new Error("The inference API did not respond correctly.");
      const modelData = await modelsResponse.json();
      const health: HealthPayload = await healthResponse.json();
      const available: ModelOption[] = Array.isArray(modelData.models) ? modelData.models : [];
      setModels(available);
      if (modelData.selected_model) setModel(modelData.selected_model);
      const isReady = health.status === "ok" && available.some((item) => item.id === modelData.selected_model && item.loaded);
      setStatus(isReady ? "ready" : "degraded");
      setStatusMessage(isReady
        ? `${modelData.models?.find((item: ModelOption) => item.id === modelData.selected_model)?.name || "Model"} and gallery are ready`
        : health.startup_error || "The model or compatible retrieval gallery is not ready. Check the backend setup.");
    } catch {
      setStatus("degraded");
      setStatusMessage("Cannot reach the API. Start the FastAPI backend, then select Check again.");
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const timer = window.setInterval(() => { void refreshStatus(); }, 15000);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  const clear = useCallback(() => {
    setQuery("");
    setFile(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
    setResults(null);
    setError("");
  }, []);

  function acceptFile(next?: File) {
    if (!next) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(next.type)) {
      setError("Please choose a JPEG, PNG, or WEBP image.");
      return;
    }
    if (next.size > 10 * 1024 * 1024) {
      setError("The image must be 10 MB or smaller.");
      return;
    }
    setFile(next);
    setPreview(URL.createObjectURL(next));
    setError("");
    setResults(null);
  }

  async function search() {
    setError("");
    setResults(null);
    if (mode === "text" && !query.trim()) return setError("Write a short description of the image you want to find.");
    if (mode === "image" && !file) return setError("Choose an image before starting the search.");
    if (status !== "ready" || !models.some((item) => item.id === model && item.loaded)) {
      return setError("The model is not ready yet. Use “Check connection” and try again.");
    }
    setLoading(true);
    try {
      let response: Response;
      if (mode === "text") {
        response = await fetch(`${API_BASE}/search/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: query.trim(), top_k: topK, model }),
        });
      } else {
        const body = new FormData();
        body.append("file", file as File);
        response = await fetch(`${API_BASE}/search/image?top_k=${topK}&model=${encodeURIComponent(model)}`, {
          method: "POST",
          body,
        });
      }
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : `Search failed (HTTP ${response.status}).`);
      }
      setResults(payload as SearchResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed. Check the API connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  function resultCaption(item: SearchResult) {
    if (item.caption) return item.caption;
    if (typeof item.result === "string") return item.result;
    if (item.result && typeof item.result === "object" && "caption" in item.result) {
      return String((item.result as { caption?: unknown }).caption ?? "");
    }
    return "";
  }

  function imageUrl(item: SearchResult) {
    if (!item.image_url) return "";
    return item.image_url.startsWith("http") ? item.image_url : `${API_BASE}${item.image_url}`;
  }

  const filteredResults = (results?.results || []).filter((item) => Number(item.similarity ?? 0) >= minSimilarity);
  const selectedModel = models.find((item) => item.id === model);

  return (
    <main className="min-h-screen overflow-hidden bg-[#f6f7fb] text-[#171a2b]">
      <div className="pointer-events-none fixed inset-x-0 top-0 -z-0 h-[560px] bg-[radial-gradient(ellipse_at_top,_rgba(121,112,232,0.13),_transparent_65%)]" />
      <header className="relative z-10 mx-auto flex max-w-[1320px] items-center justify-between px-5 py-5 sm:px-8 lg:px-12">
        <Link href="/" className="flex items-center gap-3" aria-label="VisionMatch home">
          <span className="grid h-10 w-10 place-items-center rounded-2xl bg-[#242943] text-[#d9ff70] shadow-lg shadow-[#242943]/15"><BrainCircuit size={21} /></span>
          <span className="text-base font-black tracking-tight">vision<span className="text-[#7770e8]">match</span><span className="ml-2 align-top text-[9px] font-bold uppercase tracking-widest text-[#9a9baa]">AI</span></span>
        </Link>
        <nav className="flex items-center gap-2 text-xs font-bold text-[#777b8c] sm:gap-6">
          <a href="#workspace" className="rounded-full px-3 py-2 transition hover:bg-white hover:text-[#7770e8]">Workspace</a>
          <Link href="/compare" className="rounded-full px-3 py-2 transition hover:bg-white hover:text-[#7770e8]">Model insights</Link>
          <Link href="/about" className="rounded-full px-3 py-2 transition hover:bg-white hover:text-[#7770e8]">About</Link>
        </nav>
      </header>

      <section className="relative z-10 mx-auto grid max-w-[1320px] items-center gap-12 px-5 pb-20 pt-12 sm:px-8 lg:grid-cols-[1.08fr_.92fr] lg:px-12 lg:pb-28 lg:pt-20">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-[#e7e5f4] bg-white/80 px-4 py-2 text-[10px] font-extrabold uppercase tracking-[1.8px] text-[#7770e8] shadow-sm"><Sparkles size={13} /> Cross-modal intelligence</div>
          <h1 className="mt-7 max-w-3xl text-[clamp(3.5rem,7vw,6.6rem)] font-black leading-[.94] tracking-[-.075em]">Find the image<br />behind the <span className="text-[#7770e8]">words.</span></h1>
          <p className="mt-7 max-w-xl text-base leading-8 text-[#777b8c]">Search a collection of real-world images with natural language, or upload a picture to discover the captions that describe it. One shared embedding space connects both modalities.</p>
          <div className="mt-9 flex flex-wrap items-center gap-4">
            <a href="#workspace" className="inline-flex items-center gap-3 rounded-full bg-[#242943] px-6 py-4 text-xs font-extrabold text-white shadow-xl shadow-[#242943]/15 transition hover:-translate-y-0.5 hover:bg-[#7770e8]">Open search workspace <ArrowDown size={15} /></a>
            <span className="flex items-center gap-2 text-xs font-semibold text-[#9295a4]"><Database size={15} /> Flickr30k research dataset</span>
          </div>
          <div className="mt-12 grid max-w-lg grid-cols-2 gap-3">
            <div className="rounded-2xl border border-[#e9eaf1] bg-white/75 p-4"><Search size={18} className="text-[#7770e8]" /><p className="mt-3 text-sm font-extrabold">Text → Image</p><p className="mt-1 text-[11px] leading-5 text-[#9295a4]">Describe a scene. Find matching images.</p></div>
            <div className="rounded-2xl border border-[#e9eaf1] bg-white/75 p-4"><ImageIcon size={18} className="text-[#7770e8]" /><p className="mt-3 text-sm font-extrabold">Image → Text</p><p className="mt-1 text-[11px] leading-5 text-[#9295a4]">Upload a picture. Retrieve related captions.</p></div>
          </div>
        </div>

        <div className="relative mx-auto w-full max-w-[540px]">
          <div className="absolute -inset-5 rounded-[38px] bg-gradient-to-br from-[#e6e3ff] via-transparent to-[#e8f3d4] blur-2xl" />
          <div className="relative rounded-[30px] border border-white bg-white/90 p-5 shadow-[0_30px_90px_-35px_rgba(36,41,67,.28)] backdrop-blur sm:p-7">
            <div className="flex items-center justify-between border-b border-[#eff0f5] pb-5"><div><p className="text-[10px] font-extrabold uppercase tracking-[1.8px] text-[#999cad]">How it works</p><h2 className="mt-1 text-lg font-black">A shared visual language</h2></div><span className="grid h-10 w-10 place-items-center rounded-2xl bg-[#f0efff] text-[#7770e8]"><BrainCircuit size={20} /></span></div>
            <div className="mt-6 space-y-3">
              <div className="flex items-center gap-4 rounded-2xl bg-[#f4f2ff] p-4"><span className="grid h-11 w-11 place-items-center rounded-xl bg-white text-[#7770e8]"><ImageIcon size={20} /></span><div className="min-w-0 flex-1"><p className="text-xs font-extrabold">Image encoder</p><p className="mt-1 text-[11px] text-[#85899b]">ResNet50 extracts visual features</p></div><BadgeCheck size={17} className="text-[#7770e8]" /></div>
              <div className="flex justify-center text-[#b1addf]"><ArrowDown size={18} /></div>
              <div className="rounded-2xl bg-[#242943] p-5 text-white"><div className="flex items-center justify-between"><p className="text-xs font-extrabold">Shared embedding space</p><span className="rounded-full bg-white/10 px-3 py-1 text-[9px] font-bold text-[#d9ff70]">512 dimensions</span></div><div className="mt-4 flex gap-1.5">{Array.from({ length: 18 }).map((_, i) => <span key={i} className="h-8 flex-1 rounded-md" style={{ backgroundColor: i % 3 === 0 ? "#d9ff70" : i % 3 === 1 ? "#8c84ef" : "#555d80", opacity: .45 + (i % 4) * .15 }} />)}</div><p className="mt-3 text-[10px] text-white/50">Images and language are mapped into comparable vectors.</p></div>
              <div className="flex justify-center text-[#b1addf]"><ArrowDown size={18} /></div>
              <div className="flex items-center gap-4 rounded-2xl bg-[#eff7e2] p-4"><span className="grid h-11 w-11 place-items-center rounded-xl bg-white text-[#65822f]"><Search size={20} /></span><div className="min-w-0 flex-1"><p className="text-xs font-extrabold">Similarity-based retrieval</p><p className="mt-1 text-[11px] text-[#85899b]">Rank the closest cross-modal matches</p></div><ArrowUpRight size={17} className="text-[#65822f]" /></div>
            </div>
          </div>
        </div>
      </section>

      <section id="workspace" className="relative z-10 scroll-mt-6 border-y border-[#e8e9f0] bg-white py-16 sm:py-20">
        <div className="mx-auto max-w-[1160px] px-5 sm:px-8 lg:px-12">
          <div className="flex flex-wrap items-end justify-between gap-5">
            <div><p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">Retrieval workspace</p><h2 className="mt-3 text-3xl font-black tracking-[-1.5px] sm:text-4xl">What do you want to find?</h2><p className="mt-3 max-w-xl text-sm leading-6 text-[#85899b]">Choose a search direction, provide your input, and explore the highest-ranked matches.</p></div>
            <div className={`flex items-center gap-2 rounded-full border px-3 py-2 text-[10px] font-bold ${status === "ready" ? "border-[#dce9c6] bg-[#f3f8e9] text-[#5e782f]" : status === "checking" ? "border-[#e8e7f0] bg-[#f8f8fc] text-[#777b8c]" : "border-[#f0dfcb] bg-[#fff8ef] text-[#a46b2e]"}`}>
              <span className={`h-2 w-2 rounded-full ${status === "ready" ? "bg-[#82a844]" : status === "checking" ? "animate-pulse bg-[#aaa5db]" : "bg-[#d69b55]"}`} />
              {status === "ready" ? "Inference ready" : status === "checking" ? "Checking backend" : "Backend unavailable"}
            </div>
          </div>

          <div className="mt-8 grid overflow-hidden rounded-[28px] border border-[#e8e9f0] bg-[#fbfbfd] shadow-[0_20px_65px_-45px_rgba(36,41,67,.25)] lg:grid-cols-[.72fr_1.28fr]">
            <aside className="border-b border-[#e8e9f0] bg-[#f8f8fc] p-5 sm:p-7 lg:border-b-0 lg:border-r">
              <p className="text-[10px] font-extrabold uppercase tracking-[1.7px] text-[#999cad]">01 / Search direction</p>
              <div className="mt-4 grid grid-cols-2 gap-2 lg:grid-cols-1">
                <button type="button" onClick={() => { setMode("text"); setResults(null); setError(""); }} aria-pressed={mode === "text"} className={`flex items-center gap-3 rounded-2xl border p-4 text-left transition ${mode === "text" ? "border-[#aaa3ef] bg-white shadow-sm" : "border-transparent bg-transparent hover:border-[#e7e6f0] hover:bg-white"}`}><span className={`grid h-10 w-10 place-items-center rounded-xl ${mode === "text" ? "bg-[#f0efff] text-[#7770e8]" : "bg-white text-[#9699a8]"}`}><Search size={18} /></span><span><span className="block text-xs font-extrabold">Text to Image</span><span className="mt-1 hidden text-[10px] text-[#999cad] sm:block">Search with words</span></span></button>
                <button type="button" onClick={() => { setMode("image"); setResults(null); setError(""); }} aria-pressed={mode === "image"} className={`flex items-center gap-3 rounded-2xl border p-4 text-left transition ${mode === "image" ? "border-[#aaa3ef] bg-white shadow-sm" : "border-transparent bg-transparent hover:border-[#e7e6f0] hover:bg-white"}`}><span className={`grid h-10 w-10 place-items-center rounded-xl ${mode === "image" ? "bg-[#f0efff] text-[#7770e8]" : "bg-white text-[#9699a8]"}`}><ImageIcon size={18} /></span><span><span className="block text-xs font-extrabold">Image to Text</span><span className="mt-1 hidden text-[10px] text-[#999cad] sm:block">Search with a picture</span></span></button>
              </div>

              <div className="mt-7 border-t border-[#e8e9f0] pt-6">
                <label htmlFor="model" className="flex items-center justify-between text-xs font-extrabold"><span>Inference model</span><span className="text-[9px] font-bold text-[#a0a2af]">API CONFIGURED</span></label>
                <div className="relative mt-2"><select id="model" value={model} onChange={(e) => setModel(e.target.value)} className="w-full appearance-none rounded-xl border border-[#e5e6ee] bg-white px-3 py-3 pr-9 text-xs font-semibold outline-none transition focus:border-[#9c95ec]" disabled={models.length === 0}><option value={model}>{selectedModel?.name || model}</option>{models.filter((item) => item.id !== model).map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select><ChevronDown size={15} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[#85899b]" /></div>
                <p className="mt-2 text-[10px] leading-5 text-[#999cad]">Only models with a loaded checkpoint and matching gallery are available from the running API.</p>
                <label htmlFor="topk" className="mt-5 block text-xs font-extrabold">Results to retrieve</label>
                <select id="topk" value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="mt-2 w-full rounded-xl border border-[#e5e6ee] bg-white px-3 py-3 text-xs font-semibold outline-none focus:border-[#9c95ec]">{[5, 10, 20, 50].map((n) => <option key={n} value={n}>{n} results</option>)}</select>
              </div>
              <div className="mt-6 rounded-2xl border border-[#e8e9f0] bg-white p-4">
                <div className="flex items-center justify-between gap-3"><span className="text-[10px] font-extrabold uppercase tracking-wider text-[#999cad]">API connection</span><button type="button" onClick={() => void refreshStatus()} className="inline-flex items-center gap-1.5 text-[10px] font-extrabold text-[#7770e8] hover:underline"><RefreshCw size={12} /> Check again</button></div>
                <p className="mt-2 break-words text-[11px] leading-5 text-[#777b8c]">{statusMessage}</p>
              </div>
            </aside>

            <div className="p-5 sm:p-7 lg:p-9">
              <p className="text-[10px] font-extrabold uppercase tracking-[1.7px] text-[#999cad]">02 / Your input</p>
              {mode === "text" ? <div className="mt-5">
                <label htmlFor="query" className="text-sm font-extrabold">Describe the image you have in mind</label>
                <textarea id="query" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") void search(); }} maxLength={1000} rows={7} placeholder="For example: A group of children playing football on a sunny field…" className="mt-3 w-full resize-y rounded-2xl border border-[#e5e6ee] bg-white p-5 text-sm leading-7 outline-none transition placeholder:text-[#b2b4c0] focus:border-[#9c95ec] focus:ring-4 focus:ring-[#7770e8]/10" />
                <div className="mt-2 flex justify-between text-[10px] text-[#a0a2af]"><span>Tip: include objects, actions, and scene details.</span><span>{query.length}/1000</span></div>
              </div> : <div className="mt-5">
                <p className="text-sm font-extrabold">Upload an image to find matching captions</p>
                <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={(e) => acceptFile(e.target.files?.[0])} />
                {preview ? <div className="relative mt-3 overflow-hidden rounded-2xl border border-[#e5e6ee] bg-white p-3"><Image src={preview} alt="Selected image preview" width={900} height={600} unoptimized className="mx-auto max-h-[340px] w-auto rounded-xl object-contain" /><div className="mt-3 flex items-center justify-between gap-3 border-t border-[#eff0f5] pt-3"><span className="min-w-0 truncate text-xs font-semibold text-[#777b8c]">{file?.name}</span><button type="button" onClick={() => { setFile(null); setPreview(null); if (inputRef.current) inputRef.current.value = ""; }} aria-label="Remove selected image" className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-[#fff0f0] text-[#bd5a5a] transition hover:bg-[#ffe2e2]"><X size={15} /></button></div></div> : <button type="button" onClick={() => inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); acceptFile(e.dataTransfer.files?.[0]); }} className="mt-3 flex min-h-[260px] w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-[#d9d6f1] bg-white px-5 text-center transition hover:border-[#9991e9] hover:bg-[#fbfaff]"><span className="grid h-14 w-14 place-items-center rounded-2xl bg-[#f0efff] text-[#7770e8]"><Upload size={24} /></span><span className="mt-4 text-sm font-extrabold">Drop your image here or browse</span><span className="mt-2 text-[11px] text-[#999cad]">JPEG, PNG, or WEBP · Up to 10 MB</span></button>}
              </div>}

              {error && <div role="alert" className="mt-5 flex items-start gap-3 rounded-2xl border border-[#f1d9d9] bg-[#fff7f7] p-4 text-xs leading-6 text-[#a64d4d]"><CircleAlert size={16} className="mt-0.5 shrink-0" /><span>{error}</span></div>}
              <div className="mt-7 flex flex-wrap items-center justify-between gap-3 border-t border-[#e8e9f0] pt-6">
                <button type="button" onClick={clear} className="rounded-full px-4 py-3 text-xs font-bold text-[#85899b] transition hover:bg-[#f0f0f6] hover:text-[#33384d]">Clear input</button>
                <button type="button" onClick={() => void search()} disabled={loading || status !== "ready" || !selectedModel?.loaded} className="inline-flex items-center gap-3 rounded-full bg-[#7770e8] px-7 py-4 text-xs font-extrabold text-white shadow-lg shadow-[#7770e8]/20 transition hover:-translate-y-0.5 hover:bg-[#625bd1] disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:translate-y-0">{loading ? <><LoaderCircle size={15} className="animate-spin" /> Searching…</> : <>Find matches <ArrowRight size={15} /></>}</button>
              </div>
              {status !== "ready" && <p className="mt-3 text-right text-[10px] text-[#a0a2af]">Search is enabled when the model and gallery are ready.</p>}
            </div>
          </div>

          {loading && <div className="mt-8 flex items-center justify-center gap-3 rounded-2xl border border-[#e8e9f0] bg-[#fbfbfd] p-8 text-xs font-semibold text-[#777b8c]"><LoaderCircle size={18} className="animate-spin text-[#7770e8]" /> Searching the retrieval gallery…</div>}

          {results && !loading && <section aria-live="polite" className="mt-10">
            <div className="flex flex-wrap items-end justify-between gap-4"><div><p className="text-[10px] font-extrabold uppercase tracking-[1.8px] text-[#7770e8]">03 / Retrieval results</p><h3 className="mt-2 text-2xl font-black tracking-tight">Your closest matches</h3><p className="mt-2 text-xs text-[#85899b]">{results.count} result(s) returned by {results.model}</p></div><div className="flex items-center gap-4 rounded-2xl border border-[#e8e9f0] bg-white px-4 py-3 text-[10px] font-bold text-[#85899b]"><span className="flex items-center gap-1.5"><Clock3 size={13} /> {results.elapsed_ms.toFixed(1)} ms</span><span className="h-4 w-px bg-[#e8e9f0]" /><span>{results.count} matches</span></div></div>
            {mode === "image" && preview && <div className="mt-5 flex items-center gap-4 rounded-2xl border border-[#e8e9f0] bg-white p-3"><Image src={preview} alt="Query image" width={80} height={80} unoptimized className="h-16 w-16 rounded-xl object-cover" /><div><p className="text-[10px] font-extrabold uppercase tracking-wider text-[#999cad]">Your query image</p><p className="mt-1 text-xs font-semibold text-[#55596b]">{file?.name}</p></div></div>}
            <div className="mt-5 flex flex-wrap items-center justify-between gap-3"><p className="text-[11px] text-[#85899b]">Showing {filteredResults.length} of {results.count} results</p><label className="flex items-center gap-2 text-[10px] font-bold text-[#777b8c]">Minimum similarity <select value={minSimilarity} onChange={(e) => setMinSimilarity(Number(e.target.value))} className="rounded-lg border border-[#e5e6ee] bg-white px-2.5 py-2 text-[10px] outline-none focus:border-[#9c95ec]"><option value={-1}>All scores</option><option value={0}>0.00</option><option value={0.2}>0.20</option><option value={0.4}>0.40</option><option value={0.6}>0.60</option></select></label></div>
            {filteredResults.length === 0 ? <div className="mt-5 rounded-2xl border border-dashed border-[#dedfea] bg-white p-10 text-center"><Search size={25} className="mx-auto text-[#a5a1d8]" /><p className="mt-3 text-sm font-extrabold">No matches at this threshold</p><p className="mt-1 text-xs text-[#999cad]">Lower the minimum similarity to see more retrieved items.</p></div> : <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">{filteredResults.map((item, index) => <article key={`${item.rank}-${item.image_id || index}`} className="group overflow-hidden rounded-2xl border border-[#e8e9f0] bg-white shadow-[0_12px_35px_-28px_rgba(36,41,67,.3)] transition hover:-translate-y-1 hover:shadow-lg">
              {mode === "text" && imageUrl(item) ? <div className="relative h-52 overflow-hidden bg-[#f0efff]"><Image src={imageUrl(item)} alt={item.image_id || "Retrieved Flickr30k image"} fill unoptimized sizes="(max-width: 768px) 100vw, 33vw" className="object-cover transition duration-500 group-hover:scale-[1.03]" /></div> : <div className="flex min-h-44 items-center justify-center bg-gradient-to-br from-[#f2f0ff] to-[#f6f8ed] px-6 text-center"><div><ImageIcon size={26} className="mx-auto text-[#8881df]" /><p className="mt-3 text-xs font-semibold leading-6 text-[#55596b]">{resultCaption(item) || "Caption unavailable"}</p></div></div>}
              <div className="p-4"><div className="flex items-center justify-between gap-3"><span className="rounded-full bg-[#f0efff] px-3 py-1.5 text-[9px] font-extrabold uppercase tracking-wider text-[#6e67d2]">Rank #{item.rank || index + 1}</span><span className="text-[10px] font-bold tabular-nums text-[#85899b]">Similarity {Number(item.similarity ?? 0).toFixed(4)}</span></div><p className="mt-3 break-words text-xs font-bold leading-5 text-[#55596b]">{mode === "image" ? resultCaption(item) : (item.image_id || "Retrieved image")}</p></div>
            </article>)}</div>}
          </section>}
        </div>
      </section>

      <footer className="relative z-10 mx-auto flex max-w-[1320px] flex-wrap items-center justify-between gap-4 px-5 py-8 text-[10px] text-[#999cad] sm:px-8 lg:px-12"><span>© VisionMatch AI · Academic deep learning project · Flickr30k</span><div className="flex items-center gap-5"><Link href="/about" className="font-bold transition hover:text-[#7770e8]">About</Link><Link href="/compare" className="font-bold transition hover:text-[#7770e8]">Model insights</Link><a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="font-bold transition hover:text-[#7770e8]">API docs <ArrowUpRight size={11} className="inline" /></a></div></footer>
    </main>
  );
}
