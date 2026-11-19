"use client";

import { useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import {
  ArrowDownRight, ArrowRight, ArrowUpRight, BrainCircuit, Check,
  ChevronDown, CircleHelp, Command, Compass, FileImage, Image as ImageIcon,
  Layers3, Menu, MousePointer2, Search, Sparkles, Upload, X, Zap,
} from "lucide-react";

type Mode = "text" | "image";

const gallery = [
  { title: "Quiet moments", tag: "LIFESTYLE", image: "https://images.unsplash.com/photo-1470252649378-9c29740c9fa8?w=900&q=85", className: "col-span-2 row-span-2" },
  { title: "Into the blue", tag: "NATURE", image: "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?w=800&q=85", className: "col-span-1 row-span-1" },
  { title: "City stories", tag: "URBAN", image: "https://images.unsplash.com/photo-1519608487953-e999c86e7455?w=800&q=85", className: "col-span-1 row-span-1" },
];

export default function Home() {
  const [mode, setMode] = useState<Mode>("text");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function chooseMode(next: Mode) {
    setMode(next);
    setNotice("");
  }

  function acceptFile(next?: File) {
    if (!next) return;
    if (!next.type.startsWith("image/")) {
      setNotice("Please choose an image file (PNG, JPG or WEBP).");
      return;
    }
    if (next.size > 10 * 1024 * 1024) {
      setNotice("The image must be smaller than 10 MB.");
      return;
    }
    setFile(next);
    setPreview(URL.createObjectURL(next));
    setNotice("");
  }

  function search() {
    if (mode === "text" && !query.trim()) {
      setNotice("Add a short description to get started.");
      return;
    }
    if (mode === "image" && !file) {
      setNotice("Upload an image first to continue.");
      return;
    }
    setNotice("Your input is ready. Live retrieval will be enabled when the trained model API is connected.");
  }

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#f7f7f4] text-[#171b2b]">
      <div className="pointer-events-none fixed inset-0 -z-0 opacity-[0.22] [background-image:radial-gradient(#a7a9b4_0.65px,transparent_0.65px)] [background-size:18px_18px]" />
      <header className="relative z-20 mx-auto flex max-w-[1440px] items-center justify-between px-5 py-5 sm:px-8 lg:px-14">
        <Link href="/" className="group flex items-center gap-3">
          <span className="grid h-11 w-11 place-items-center rounded-2xl bg-[#242943] text-[#d9ff70] shadow-lg shadow-slate-300/50 transition group-hover:-rotate-6"><BrainCircuit size={23} /></span>
          <span><span className="block text-[17px] font-black tracking-[-0.8px]">vision<span className="text-[#7770e8]">match</span></span><span className="block text-[9px] font-bold uppercase tracking-[2.8px] text-[#8a8d9a]">Cross-modal intelligence</span></span>
        </Link>
        <nav className="hidden items-center gap-9 text-[13px] font-semibold text-[#696d7b] md:flex">
          <a href="#explore" className="transition hover:text-[#7770e8]">Explore</a>
          <a href="#how-it-works" className="transition hover:text-[#7770e8]">How it works</a>
          <Link href="/compare" className="transition hover:text-[#7770e8]">Model insights</Link>
        </nav>
        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-2 rounded-full border border-[#e7e6df] bg-white/80 px-3 py-2 text-[10px] font-bold text-[#777b88] sm:flex"><span className="h-1.5 w-1.5 rounded-full bg-[#e5a34b]" /> DEVELOPMENT PREVIEW</span>
          <button onClick={() => setMobileOpen(!mobileOpen)} className="rounded-xl border border-[#e6e5df] bg-white p-2.5 md:hidden" aria-label="Toggle menu">{mobileOpen ? <X size={18}/> : <Menu size={18}/>}</button>
          <a href="#explore" className="hidden items-center gap-2 rounded-full bg-[#242943] px-5 py-3 text-xs font-bold text-white transition hover:bg-[#7770e8] sm:flex">Try it now <ArrowUpRight size={14}/></a>
        </div>
      </header>
      {mobileOpen && <nav className="relative z-20 mx-5 mb-3 flex flex-col gap-4 rounded-2xl border border-[#e9e7df] bg-white p-5 text-sm font-semibold md:hidden"><a href="#explore" onClick={() => setMobileOpen(false)}>Explore</a><a href="#how-it-works" onClick={() => setMobileOpen(false)}>How it works</a><Link href="/compare">Model insights</Link></nav>}

      <section className="relative z-10 mx-auto grid max-w-[1440px] items-center gap-12 px-5 pb-16 pt-10 sm:px-8 md:pt-16 lg:grid-cols-[1.02fr_0.98fr] lg:px-14 lg:pb-24">
        <div className="relative z-10">
          <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-[#e6e4dc] bg-white/80 px-4 py-2 text-[10px] font-extrabold uppercase tracking-[1.8px] text-[#7770e8] shadow-sm"><Sparkles size={13}/> A new way to find what you see</div>
          <h1 className="max-w-[720px] text-[clamp(3.5rem,7vw,6.8rem)] font-black leading-[0.91] tracking-[-0.075em]">Meaning is<br/> <span className="relative inline-block text-[#7770e8]">visual.</span> Find it<span className="text-[#d4ed75]">.</span></h1>
          <p className="mt-8 max-w-[490px] text-[15px] leading-8 text-[#777b88] sm:text-[17px]">One search space. Two directions. Discover the images behind your words — or the words that bring an image to life.</p>
          <div className="mt-9 flex flex-wrap items-center gap-4">
            <a href="#explore" className="group inline-flex items-center gap-4 rounded-full bg-[#242943] py-2 pl-6 pr-2 text-sm font-bold text-white shadow-xl shadow-[#242943]/15 transition hover:-translate-y-1"><span>Explore the workspace</span><span className="grid h-10 w-10 place-items-center rounded-full bg-[#d9ff70] text-[#242943] transition group-hover:rotate-45"><ArrowUpRight size={19}/></span></a>
            <span className="text-[11px] font-semibold text-[#999ba5]">Built for image ↔ language retrieval</span>
          </div>
          <div className="mt-14 flex flex-wrap gap-x-8 gap-y-4 border-t border-[#e5e3dc] pt-6">
            <div><p className="text-xl font-black tracking-tight">01 <span className="text-[#7770e8]">/</span> 02</p><p className="mt-1 text-[9px] font-bold uppercase tracking-[1.7px] text-[#9b9da6]">Search directions</p></div>
            <div><p className="text-xl font-black tracking-tight">AI <span className="text-[#7770e8]">+</span> Vision</p><p className="mt-1 text-[9px] font-bold uppercase tracking-[1.7px] text-[#9b9da6]">Deep learning core</p></div>
            <div><p className="text-xl font-black tracking-tight">Flickr30k</p><p className="mt-1 text-[9px] font-bold uppercase tracking-[1.7px] text-[#9b9da6]">Research dataset</p></div>
          </div>
        </div>

        <div className="relative mx-auto h-[420px] w-full max-w-[620px] sm:h-[520px]">
          <div className="absolute right-[3%] top-[4%] h-[78%] w-[72%] rotate-[5deg] rounded-[34px] bg-[#d9ff70]" />
          <div className="absolute right-[7%] top-[1%] h-[80%] w-[72%] overflow-hidden rounded-[30px] border-[8px] border-white bg-[#d7d8d2] shadow-2xl shadow-[#222943]/15">
            <div className="absolute inset-0 bg-cover bg-center" style={{backgroundImage:"url('https://images.unsplash.com/photo-1470252649378-9c29740c9fa8?w=1200&q=90')"}}/>
            <div className="absolute inset-0 bg-gradient-to-t from-[#111827]/65 via-transparent to-transparent"/>
            <div className="absolute bottom-6 left-6 right-6 text-white"><span className="rounded-full border border-white/30 bg-white/15 px-3 py-1 text-[9px] font-bold uppercase tracking-[1.6px] backdrop-blur">Visual concept</span><p className="mt-3 text-2xl font-bold tracking-tight">Chasing the golden hour.</p></div>
          </div>
          <div className="absolute bottom-[4%] left-[1%] w-[54%] rounded-[24px] border border-white/80 bg-white/90 p-4 shadow-2xl shadow-[#242943]/10 backdrop-blur-xl sm:p-5">
            <div className="flex items-center justify-between"><span className="text-[9px] font-extrabold uppercase tracking-[1.6px] text-[#9295a0]">Semantic match</span><span className="flex items-center gap-1 rounded-full bg-[#eff8d9] px-2 py-1 text-[9px] font-bold text-[#647d2b]"><Sparkles size={10}/> Concept</span></div>
            <div className="mt-4 flex items-center gap-3"><span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[#eeeaff] text-[#7770e8]"><BrainCircuit size={19}/></span><div className="min-w-0"><p className="truncate text-xs font-bold">A peaceful landscape at sunset</p><p className="mt-1 text-[10px] text-[#999ba5]">Text ↔ image embedding</p></div></div>
            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-[#eeedf0]"><div className="h-full w-[78%] rounded-full bg-[#7770e8]"/></div>
          </div>
          <div className="absolute right-[1%] top-[58%] grid h-[78px] w-[78px] place-items-center rounded-[24px] border-[5px] border-white bg-[#242943] text-[#d9ff70] shadow-xl sm:h-[94px] sm:w-[94px]"><ImageIcon size={34}/></div>
          <div className="absolute left-[10%] top-[5%] grid h-12 w-12 place-items-center rounded-2xl bg-white text-[#7770e8] shadow-xl"><Sparkles size={20}/></div>
        </div>
      </section>

      <section id="explore" className="relative z-10 border-y border-[#e9e7df] bg-white/75 py-16 sm:py-20">
        <div className="mx-auto max-w-[1180px] px-5 sm:px-8">
          <div className="mb-9 flex flex-wrap items-end justify-between gap-5">
            <div><p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">Your creative search space</p><h2 className="mt-3 text-3xl font-black tracking-[-1.8px] sm:text-4xl">What do you have in mind?</h2><p className="mt-3 text-sm text-[#898c98]">Start with a thought or drop in a visual. We'll take it from there.</p></div>
            <div className="hidden items-center gap-2 text-[10px] font-bold uppercase tracking-[1.4px] text-[#9b9da6] sm:flex"><Command size={13}/> Interactive demo interface</div>
          </div>
          <div className="grid overflow-hidden rounded-[28px] border border-[#e9e7df] bg-[#fbfbf9] shadow-[0_22px_80px_-45px_rgba(30,35,60,0.25)] lg:grid-cols-[0.78fr_1.22fr]">
            <div className="border-b border-[#e9e7df] p-6 sm:p-9 lg:border-b-0 lg:border-r">
              <p className="text-[10px] font-extrabold uppercase tracking-[1.8px] text-[#9a9ca6]">01 / Choose a direction</p>
              <div className="mt-6 space-y-3">
                <button onClick={() => chooseMode("text")} className={`flex w-full items-center gap-4 rounded-2xl border p-4 text-left transition ${mode === "text" ? "border-[#7770e8] bg-[#f1efff] shadow-sm" : "border-[#e9e7df] bg-white hover:border-[#c8c5ef]"}`}>
                  <span className={`grid h-12 w-12 place-items-center rounded-xl ${mode === "text" ? "bg-[#7770e8] text-white" : "bg-[#f2f1ed] text-[#747887]"}`}><Search size={20}/></span><span className="min-w-0 flex-1"><span className="block text-sm font-extrabold">Text to Image</span><span className="mt-1 block text-[11px] text-[#898c98]">Find visuals with words</span></span>{mode === "text" && <Check size={17} className="text-[#7770e8]"/>}
                </button>
                <button onClick={() => chooseMode("image")} className={`flex w-full items-center gap-4 rounded-2xl border p-4 text-left transition ${mode === "image" ? "border-[#7770e8] bg-[#f1efff] shadow-sm" : "border-[#e9e7df] bg-white hover:border-[#c8c5ef]"}`}>
                  <span className={`grid h-12 w-12 place-items-center rounded-xl ${mode === "image" ? "bg-[#7770e8] text-white" : "bg-[#f2f1ed] text-[#747887]"}`}><ImageIcon size={20}/></span><span className="min-w-0 flex-1"><span className="block text-sm font-extrabold">Image to Text</span><span className="mt-1 block text-[11px] text-[#898c98]">Find language from visuals</span></span>{mode === "image" && <Check size={17} className="text-[#7770e8]"/>}
                </button>
              </div>
              <div className="mt-8 rounded-2xl bg-[#242943] p-5 text-white"><div className="flex items-center gap-2 text-[#d9ff70]"><Zap size={15}/><span className="text-[10px] font-extrabold uppercase tracking-[1.5px]">Under the hood</span></div><p className="mt-3 text-sm font-bold">A shared semantic space.</p><p className="mt-2 text-xs leading-6 text-white/55">Visual and language features are mapped into comparable embeddings by the retrieval model.</p><div className="mt-4 flex items-center gap-2 text-[10px] font-bold text-white/75"><span className="h-px w-6 bg-[#d9ff70]"/> CNN + LSTM architecture</div></div>
            </div>
            <div className="p-6 sm:p-9">
              <div className="flex items-center justify-between"><p className="text-[10px] font-extrabold uppercase tracking-[1.8px] text-[#9a9ca6]">02 / Your query</p><span className="rounded-full bg-[#eff8d9] px-3 py-1.5 text-[9px] font-extrabold uppercase tracking-[1.2px] text-[#647d2b]">Ready for input</span></div>
              {mode === "text" ? <div className="mt-6"><label htmlFor="query" className="mb-3 block text-sm font-bold">Describe what you're looking for</label><textarea id="query" value={query} onChange={e => {setQuery(e.target.value);setNotice("");}} rows={6} placeholder="A small dog running through a field of wildflowers on a sunny afternoon..." className="w-full resize-none rounded-2xl border border-[#e6e4dc] bg-white p-5 text-sm leading-7 outline-none transition placeholder:text-[#b1b2b9] focus:border-[#7770e8] focus:ring-4 focus:ring-[#7770e8]/10"/><div className="mt-2 flex justify-between text-[10px] text-[#a1a3ad]"><span>Natural language works best</span><span>{query.length} / 500</span></div></div> : <div className="mt-6"><p className="mb-3 text-sm font-bold">Upload your reference image</p><input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={e => acceptFile(e.target.files?.[0])}/>{preview ? <div className="relative flex min-h-[230px] items-center justify-center overflow-hidden rounded-2xl border border-[#e6e4dc] bg-white p-3"><Image src={preview} alt="Uploaded preview" width={700} height={420} unoptimized className="max-h-[260px] w-auto rounded-xl object-contain"/><button onClick={clearFile} aria-label="Remove image" className="absolute right-5 top-5 grid h-9 w-9 place-items-center rounded-full bg-[#242943] text-white"><X size={16}/></button></div> : <button onClick={() => inputRef.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault();acceptFile(e.dataTransfer.files?.[0]);}} className="flex min-h-[230px] w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-[#dcd9ee] bg-white px-5 text-center transition hover:border-[#7770e8] hover:bg-[#faf9ff]"><span className="grid h-14 w-14 place-items-center rounded-2xl bg-[#f1efff] text-[#7770e8]"><Upload size={23}/></span><span className="mt-4 text-sm font-bold">Drop an image here, or browse</span><span className="mt-2 text-[11px] text-[#9b9da6]">PNG, JPG or WEBP · Up to 10 MB</span></button>}</div>}
              {notice && <p role="status" className="mt-4 rounded-xl border border-[#e6e2ff] bg-[#f4f2ff] px-4 py-3 text-xs leading-5 text-[#5f57bd]">{notice}</p>}
              <div className="mt-7 flex flex-wrap items-center justify-between gap-4 border-t border-[#e9e7df] pt-6"><p className="flex items-center gap-2 text-[10px] text-[#9698a2]"><Sparkles size={13} className="text-[#7770e8]"/> Semantic retrieval · Flickr30k</p><button onClick={search} className="group inline-flex items-center gap-4 rounded-full bg-[#7770e8] py-2 pl-6 pr-2 text-xs font-extrabold text-white shadow-lg shadow-[#7770e8]/20 transition hover:-translate-y-0.5 hover:bg-[#5f58cc]"><span>Start searching</span><span className="grid h-9 w-9 place-items-center rounded-full bg-white text-[#7770e8] transition group-hover:rotate-45"><ArrowRight size={16}/></span></button></div>
            </div>
          </div>
          <p className="mt-4 text-center text-[10px] text-[#a4a5ad]">This is the frontend preview. Search requests will connect to the trained retrieval model in the backend integration phase.</p>
        </div>
      </section>

      <section id="how-it-works" className="relative z-10 mx-auto grid max-w-[1440px] gap-12 px-5 py-20 sm:px-8 lg:grid-cols-[0.8fr_1.2fr] lg:px-14 lg:py-28">
        <div><p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">A little more than search</p><h2 className="mt-4 max-w-[450px] text-4xl font-black leading-[1.02] tracking-[-2px] sm:text-5xl">Two languages.<br/><span className="text-[#7770e8]">One meaning.</span></h2><p className="mt-5 max-w-[390px] text-sm leading-7 text-[#858895]">VisionMatch learns relationships between pictures and their descriptions, making it possible to search across modalities.</p><Link href="/compare" className="mt-7 inline-flex items-center gap-2 text-xs font-extrabold text-[#242943] transition hover:text-[#7770e8]">Explore model insights <ArrowUpRight size={15}/></Link></div>
        <div className="grid gap-4 sm:grid-cols-2">
          <article className="rounded-[26px] border border-[#e9e7df] bg-white p-7"><span className="grid h-12 w-12 place-items-center rounded-2xl bg-[#f0efff] text-[#7770e8]"><Search size={21}/></span><p className="mt-8 text-[10px] font-extrabold uppercase tracking-[1.7px] text-[#a0a1aa]">Direction 01</p><h3 className="mt-2 text-xl font-black tracking-tight">Words → Images</h3><p className="mt-3 text-xs leading-6 text-[#898c98]">Describe a scene in your own words and retrieve visually relevant images from the collection.</p><div className="mt-7 flex items-center gap-2 text-[10px] font-bold text-[#7770e8]">Text-to-image retrieval <ArrowUpRight size={13}/></div></article>
          <article className="rounded-[26px] border border-[#e9e7df] bg-white p-7"><span className="grid h-12 w-12 place-items-center rounded-2xl bg-[#eff8d9] text-[#6b8532]"><ImageIcon size={21}/></span><p className="mt-8 text-[10px] font-extrabold uppercase tracking-[1.7px] text-[#a0a1aa]">Direction 02</p><h3 className="mt-2 text-xl font-black tracking-tight">Images → Words</h3><p className="mt-3 text-xs leading-6 text-[#898c98]">Use an image as a query and find the captions that best describe its visual content.</p><div className="mt-7 flex items-center gap-2 text-[10px] font-bold text-[#6b8532]">Image-to-text retrieval <ArrowUpRight size={13}/></div></article>
        </div>
      </section>

      <section className="relative z-10 bg-[#242943] py-16 text-white sm:py-20">
        <div className="mx-auto grid max-w-[1440px] gap-10 px-5 sm:px-8 lg:grid-cols-[0.72fr_1.28fr] lg:items-center lg:px-14">
          <div><p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#d9ff70]">Visual discovery</p><h2 className="mt-4 text-3xl font-black tracking-[-1.5px] sm:text-4xl">Every image has<br/>a story to find.</h2><p className="mt-4 max-w-[350px] text-sm leading-7 text-white/55">A glimpse of the visual world our cross-modal search is designed to explore.</p></div>
          <div className="grid h-[270px] grid-cols-3 grid-rows-2 gap-3 sm:h-[340px]">{gallery.map((item) => <div key={item.title} className={`group relative overflow-hidden rounded-2xl ${item.className}`}><div className="absolute inset-0 bg-cover bg-center transition duration-700 group-hover:scale-105" style={{backgroundImage:`url('${item.image}')`}}/><div className="absolute inset-0 bg-gradient-to-t from-[#121629]/75 via-transparent to-transparent"/><div className="absolute bottom-4 left-4"><p className="text-[8px] font-extrabold tracking-[1.7px] text-[#d9ff70]">{item.tag}</p><p className="mt-1 text-sm font-bold sm:text-base">{item.title}</p></div></div>)}</div>
        </div>
      </section>

      <footer className="relative z-10 bg-[#f7f7f4] px-5 py-8 sm:px-8 lg:px-14"><div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-5 border-t border-[#e5e3dc] pt-7"><Link href="/" className="flex items-center gap-2 text-sm font-black tracking-tight"><BrainCircuit size={18} className="text-[#7770e8]"/> visionmatch</Link><p className="text-[10px] text-[#9b9da6]">An academic deep learning project · Flickr30k</p><Link href="/compare" className="flex items-center gap-2 text-[10px] font-bold text-[#7770e8]">Model insights <ArrowUpRight size={13}/></Link></div></footer>
    </main>
  );
}
