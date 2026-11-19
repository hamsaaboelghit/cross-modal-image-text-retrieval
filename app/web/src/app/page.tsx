"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import {
  ArrowRight, ArrowUpRight, BrainCircuit, Image as ImageIcon,
  LoaderCircle, Search, Sparkles, Upload, X,
} from "lucide-react";

type Mode = "text" | "image";
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
// Search requests are sent to the configured FastAPI inference service.
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export default function Home() {
  const [mode, setMode] = useState<Mode>("text");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [topK, setTopK] = useState(5);
  const [minSimilarity, setMinSimilarity] = useState(-1);
  const [model, setModel] = useState("baseline");
  const [models, setModels] = useState<Array<{id:string;name:string;loaded:boolean}>>([]);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    Promise.all([
      fetch(`${API_BASE}/models`).then(async response => {
        if (!response.ok) throw new Error("Model service is not available.");
        return response.json();
      }),
      fetch(`${API_BASE}/health`).then(async response => {
        if (!response.ok) throw new Error("Backend health check failed.");
        return response.json();
      }),
    ])
      .then(([data, health]) => {
        if (!active) return;
        const available = Array.isArray(data.models) ? data.models : [];
        setModels(available);
        if (data.selected_model) setModel(data.selected_model);
        if (health.status !== "ok") {
          setError(health.startup_error || "The inference model or compatible gallery is not ready. Check the backend logs.");
        }
      })
      .catch(() => {
        if (active) setError("Backend is not connected yet. Start the FastAPI server and check its /health endpoint.");
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  function acceptFile(next?: File) {
    if (!next) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(next.type)) {
      setError("Choose a JPEG, PNG, or WEBP image.");
      return;
    }
    if (next.size > 10 * 1024 * 1024) {
      setError("The image must be 10 MB or smaller.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(next);
    setPreview(URL.createObjectURL(next));
    setError("");
    setResults(null);
  }

  function clear() {
    setQuery("");
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
    setResults(null);
    setError("");
  }

  async function search() {
    setError("");
    setResults(null);
    if (mode === "text" && !query.trim()) return setError("Enter a description before searching.");
    if (mode === "image" && !file) return setError("Upload an image before searching.");
    if (!models.some(item => item.id === model && item.loaded)) {
      return setError("The configured model or its compatible retrieval gallery is not ready. Check the backend /health endpoint and server logs.");
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
        const detail = typeof payload.detail === "string" ? payload.detail : "Search request failed. Check the API server and try again.";
        throw new Error(detail);
      }
      setResults(payload as SearchResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the VisionMatch API.");
    } finally {
      setLoading(false);
    }
  }

  function resultCaption(item: SearchResult) {
    if (item.caption) return item.caption;
    if (typeof item.result === "string") return item.result;
    if (item.result && typeof item.result === "object" && "caption" in item.result) {
      return String((item.result as {caption?: unknown}).caption ?? "");
    }
    return "";
  }

  function imageUrl(item: SearchResult) {
    if (!item.image_url) return "";
    return item.image_url.startsWith("http") ? item.image_url : `${API_BASE}${item.image_url}`;
  }

  const filteredResults = (results?.results || []).filter(
    item => Number(item.similarity ?? 0) >= minSimilarity
  );

  return (
    <main className="min-h-screen bg-[#f7f7f4] text-[#171b2b]">
      <header className="mx-auto flex max-w-[1280px] items-center justify-between px-5 py-6 sm:px-8">
        <Link href="/" className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-[#242943] text-[#d9ff70]"><BrainCircuit size={21}/></span><span className="text-sm font-black">vision<span className="text-[#7770e8]">match</span></span></Link>
        <nav className="flex items-center gap-5 text-xs font-bold text-[#696d7b]"><a href="#search">Search</a><Link href="/compare">Model insights</Link><Link href="/about">About</Link></nav>
      </header>
      <section className="mx-auto grid max-w-[1280px] gap-10 px-5 pb-14 pt-8 sm:px-8 lg:grid-cols-2 lg:items-center">
        <div><p className="inline-flex items-center gap-2 rounded-full border border-[#e6e4dc] bg-white px-4 py-2 text-[10px] font-extrabold uppercase tracking-widest text-[#7770e8]"><Sparkles size={13}/> Cross-modal intelligence</p><h1 className="mt-7 text-5xl font-black leading-[0.98] tracking-[-3px] sm:text-7xl">Meaning is<br/><span className="text-[#7770e8]">visual.</span> Find it.</h1><p className="mt-6 max-w-lg text-sm leading-7 text-[#777b88]">Search images with natural language, or find captions that describe an image. VisionMatch maps visual and language features into a shared semantic space.</p><a href="#search" className="mt-8 inline-flex items-center gap-3 rounded-full bg-[#242943] px-6 py-3 text-xs font-bold text-white">Start searching <ArrowRight size={15}/></a></div>
        <div className="rounded-[30px] border border-[#e9e7df] bg-white p-8 shadow-xl shadow-slate-200/50"><div className="flex items-center gap-3 text-[#7770e8]"><BrainCircuit size={24}/><span className="text-xs font-extrabold uppercase tracking-widest">One shared embedding space</span></div><div className="mt-8 grid grid-cols-2 gap-4"><div className="rounded-2xl bg-[#f1efff] p-5"><Search size={22}/><p className="mt-5 font-black">Text → Image</p><p className="mt-2 text-xs leading-5 text-[#777b88]">Describe a scene and retrieve relevant images.</p></div><div className="rounded-2xl bg-[#eff8d9] p-5"><ImageIcon size={22}/><p className="mt-5 font-black">Image → Text</p><p className="mt-2 text-xs leading-5 text-[#777b88]">Upload a picture and retrieve matching captions.</p></div></div></div>
      </section>
      <section id="search" className="border-y border-[#e9e7df] bg-white py-14">
        <div className="mx-auto max-w-[1080px] px-5 sm:px-8">
          <p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">Retrieval workspace</p><h2 className="mt-3 text-3xl font-black tracking-tight">What are you looking for?</h2>
          <div className="mt-7 grid gap-8 rounded-[26px] border border-[#e9e7df] bg-[#fbfbf9] p-5 sm:p-8 lg:grid-cols-[0.7fr_1.3fr]">
            <div><p className="text-[10px] font-extrabold uppercase tracking-widest text-[#999ba5]">Choose direction</p><div className="mt-4 space-y-3">
              <button onClick={() => {setMode("text");setResults(null);setError("");}} aria-pressed={mode==="text"} className={`flex w-full items-center gap-3 rounded-xl border p-4 text-left text-sm font-bold ${mode==="text"?"border-[#7770e8] bg-[#f1efff]":"border-[#e9e7df] bg-white"}`}><Search size={18}/> Text to Image</button>
              <button onClick={() => {setMode("image");setResults(null);setError("");}} aria-pressed={mode==="image"} className={`flex w-full items-center gap-3 rounded-xl border p-4 text-left text-sm font-bold ${mode==="image"?"border-[#7770e8] bg-[#f1efff]":"border-[#e9e7df] bg-white"}`}><ImageIcon size={18}/> Image to Text</button>
            </div>
            <label className="mt-6 block text-xs font-bold" htmlFor="model">Model</label><select id="model" value={model} onChange={e=>setModel(e.target.value)} className="mt-2 w-full rounded-xl border border-[#e5e3dc] bg-white p-3 text-sm" disabled={models.length===0}><option value={model}>{models.find(x=>x.id===model)?.name || model}</option>{models.filter(x=>x.id!==model).map(x=><option key={x.id} value={x.id}>{x.name}</option>)}</select><p className="mt-2 text-[10px] text-[#9698a2]">{models.find(x=>x.id===model)?.loaded ? "Model and compatible gallery are ready" : "Model or compatible gallery is not ready. Search is disabled until the backend is healthy."}</p>
            <label className="mt-5 block text-xs font-bold" htmlFor="topk">Number of results</label><select id="topk" value={topK} onChange={e=>setTopK(Number(e.target.value))} className="mt-2 w-full rounded-xl border border-[#e5e3dc] bg-white p-3 text-sm">{[5,10,20,50].map(n=><option key={n} value={n}>{n} results</option>)}</select></div>
            <div>{mode==="text" ? <><label htmlFor="query" className="block text-xs font-bold">Describe your image</label><textarea id="query" value={query} onChange={e=>setQuery(e.target.value)} maxLength={1000} rows={7} placeholder="A small dog running through a field of wildflowers..." className="mt-3 w-full resize-y rounded-xl border border-[#e5e3dc] bg-white p-4 text-sm leading-6 outline-none focus:border-[#7770e8]"/><p className="mt-2 text-right text-[10px] text-[#999ba5]">{query.length}/1000</p></> : <><p className="text-xs font-bold">Upload an image</p><input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={e=>acceptFile(e.target.files?.[0])}/>{preview ? <div className="relative mt-3 flex min-h-52 items-center justify-center rounded-xl border bg-white p-3"><Image src={preview} alt="Selected upload preview" width={800} height={500} unoptimized className="max-h-64 max-w-full rounded-lg object-contain"/><button onClick={clear} aria-label="Clear image" className="absolute right-3 top-3 rounded-full bg-[#242943] p-2 text-white"><X size={15}/></button></div> : <button onClick={()=>inputRef.current?.click()} className="mt-3 flex min-h-52 w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-[#dcd9ee] bg-white text-sm font-bold"><Upload size={25} className="mb-3 text-[#7770e8]"/>Choose image (JPEG, PNG, WEBP · max 10 MB)</button>}</>}
              {error && <p role="alert" className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-700">{error}</p>}
              <div className="mt-6 flex flex-wrap justify-end gap-3"><button onClick={clear} className="rounded-full border border-[#e5e3dc] bg-white px-5 py-3 text-xs font-bold">Clear</button><button onClick={search} disabled={loading || !models.some(item => item.id === model && item.loaded)} className="inline-flex items-center gap-2 rounded-full bg-[#7770e8] px-6 py-3 text-xs font-extrabold text-white disabled:cursor-not-allowed disabled:opacity-60">{loading?<><LoaderCircle size={15} className="animate-spin"/>Searching...</>:<>Search <ArrowRight size={15}/></>}</button></div>
            </div>
          </div>
          {results && <div className="mt-9"><div className="flex flex-wrap items-end justify-between gap-3"><div><h3 className="text-xl font-black">Search results</h3><p className="mt-1 text-xs text-[#858895]">{filteredResults.length} shown / {results.count} retrieved · {results.model} · {results.elapsed_ms} ms</p></div><label className="text-xs font-bold">Minimum similarity<select value={minSimilarity} onChange={e=>setMinSimilarity(Number(e.target.value))} className="ml-2 rounded-xl border border-[#e5e3dc] bg-white p-2.5 text-xs"><option value={-1}>All scores</option><option value={0}>0.00</option><option value={0.2}>0.20</option><option value={0.4}>0.40</option><option value={0.6}>0.60</option></select></label></div>{filteredResults.length===0 ? <p className="mt-5 rounded-xl bg-[#f7f7f4] p-6 text-sm text-[#858895]">No retrieved results meet this similarity threshold.</p> : <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{filteredResults.map((item,index)=><article key={`${item.rank}-${item.image_id||index}`} className="overflow-hidden rounded-2xl border border-[#e9e7df] bg-white">{mode==="text" && imageUrl(item) ? <Image src={imageUrl(item)} alt={item.image_id||"Retrieved Flickr30k image"} width={640} height={360} unoptimized className="h-48 w-full object-cover"/> : <div className="flex h-48 items-center justify-center bg-[#f1efff] text-[#7770e8]"><ImageIcon size={42}/></div>}<div className="p-4"><div className="flex items-center justify-between text-[10px] font-extrabold uppercase tracking-wider text-[#999ba5]"><span>Rank #{item.rank||index+1}</span><span>Similarity {Number(item.similarity ?? 0).toFixed(4)}</span></div><p className="mt-3 break-words text-sm font-bold">{mode==="image"?resultCaption(item):(item.image_id||"Retrieved image")}</p></div></article>)}</div>}</div>}
        </div>
      </section>
      <footer className="mx-auto flex max-w-[1280px] flex-wrap items-center justify-between gap-4 px-5 py-8 text-[10px] text-[#9698a2] sm:px-8"><span>VisionMatch AI · Flickr30k research project</span><Link href="/compare" className="inline-flex items-center gap-2 font-bold text-[#7770e8]">Model insights <ArrowUpRight size={13}/></Link></footer>
    </main>
  );
}
