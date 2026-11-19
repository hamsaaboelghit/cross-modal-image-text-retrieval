"use client";

import Link from "next/link";
import { ArrowLeft, ArrowRight, ArrowUpRight, BrainCircuit, Camera, Database, Image as ImageIcon, Languages, Network, Search, Sparkles } from "lucide-react";

const stages = [
  { number: "01", title: "Visual understanding", detail: "A ResNet50 image encoder extracts meaningful visual features from each image.", icon: Camera },
  { number: "02", title: "Language understanding", detail: "The configured text encoder processes caption sequences and learns how words combine into meaning.", icon: Languages },
  { number: "03", title: "Shared embedding space", detail: "Both modalities are projected into a comparable feature space.", icon: Network },
  { number: "04", title: "Cross-modal retrieval", detail: "The system ranks the most relevant items using similarity between embeddings.", icon: Search },
];

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-[#f7f7f4] text-[#171b2b]">
      <header className="mx-auto flex max-w-[1280px] items-center justify-between px-5 py-6 sm:px-8 lg:px-12">
        <Link href="/" className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-[#242943] text-[#d9ff70]"><BrainCircuit size={21}/></span><span className="text-sm font-black tracking-tight">vision<span className="text-[#7770e8]">match</span></span></Link>
        <Link href="/" className="flex items-center gap-2 rounded-full border border-[#e6e4dc] bg-white px-4 py-2.5 text-xs font-bold text-[#6e7180] transition hover:border-[#7770e8] hover:text-[#7770e8]"><ArrowLeft size={14}/> Back to workspace</Link>
      </header>

      <section className="mx-auto max-w-[1280px] px-5 pb-16 pt-12 sm:px-8 lg:px-12 lg:pt-20">
        <div className="max-w-3xl">
          <p className="flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]"><Sparkles size={13}/> Behind the intelligence</p>
          <h1 className="mt-5 text-5xl font-black leading-[0.98] tracking-[-3px] sm:text-7xl">Images speak.<br/><span className="text-[#7770e8]">We connect.</span></h1>
          <p className="mt-7 max-w-2xl text-base leading-8 text-[#858895]">VisionMatch is a cross-modal retrieval project that learns the relationship between images and natural-language descriptions. Search in either direction: start with words to find images, or start with an image to find relevant captions.</p>
          <div className="mt-9 flex flex-wrap gap-3"><Link href="/#search" className="inline-flex items-center gap-3 rounded-full bg-[#242943] px-6 py-3.5 text-xs font-extrabold text-white transition hover:bg-[#7770e8]">Try the workspace <ArrowUpRight size={15}/></Link><Link href="/compare" className="inline-flex items-center gap-3 rounded-full border border-[#e6e4dc] bg-white px-6 py-3.5 text-xs font-extrabold text-[#55596b] transition hover:border-[#7770e8] hover:text-[#7770e8]">View experiments <ArrowRight size={15}/></Link></div>
        </div>
      </section>

      <section className="border-y border-[#e9e7df] bg-white/80 py-16 sm:py-20">
        <div className="mx-auto max-w-[1280px] px-5 sm:px-8 lg:px-12">
          <div className="mb-10 max-w-xl"><p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">The pipeline</p><h2 className="mt-3 text-3xl font-black tracking-[-1.5px] sm:text-4xl">From two modalities<br/>to one meaning.</h2></div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">{stages.map((stage) => <article key={stage.number} className="rounded-[24px] border border-[#e9e7df] bg-[#fbfbf9] p-6 transition hover:-translate-y-1 hover:shadow-xl hover:shadow-[#242943]/5"><div className="flex items-center justify-between"><span className="text-xs font-black text-[#7770e8]">{stage.number}</span><span className="grid h-11 w-11 place-items-center rounded-2xl bg-[#f0efff] text-[#7770e8]"><stage.icon size={20}/></span></div><h3 className="mt-8 text-lg font-black tracking-tight">{stage.title}</h3><p className="mt-3 text-xs leading-6 text-[#898c98]">{stage.detail}</p></article>)}</div>
          <div className="mt-6 rounded-[26px] bg-[#242943] p-7 text-white sm:p-9">
            <div className="flex flex-wrap items-center justify-between gap-5"><div><p className="text-[10px] font-extrabold uppercase tracking-[1.7px] text-[#d9ff70]">The model concept</p><h3 className="mt-2 text-2xl font-black">Visual encoder + text encoder</h3><p className="mt-2 max-w-xl text-xs leading-6 text-white/55">The configured visual and language encoders map images and captions into a shared embedding space, where matching pairs are trained to have similar representations.</p></div><div className="flex items-center gap-2 text-[#d9ff70]"><ImageIcon size={25}/><span className="text-xl">↔</span><Languages size={25}/></div></div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-[1280px] gap-8 px-5 py-16 sm:px-8 lg:grid-cols-[0.8fr_1.2fr] lg:px-12 lg:py-20">
        <div><p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">Research foundation</p><h2 className="mt-3 text-3xl font-black tracking-[-1.5px]">Built around<br/>Flickr30k.</h2><p className="mt-4 text-sm leading-7 text-[#858895]">The project uses the public Flickr30k image-caption dataset to learn and evaluate relationships between real-world images and their natural-language descriptions.</p></div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-[24px] border border-[#e9e7df] bg-white p-6"><Database size={21} className="text-[#7770e8]"/><p className="mt-5 text-lg font-black">Flickr30k</p><p className="mt-2 text-xs leading-6 text-[#898c98]">A paired image-and-caption benchmark used for cross-modal retrieval research.</p></div>
          <div className="rounded-[24px] border border-[#e9e7df] bg-white p-6"><Network size={21} className="text-[#7770e8]"/><p className="mt-5 text-lg font-black">Two retrieval tasks</p><p className="mt-2 text-xs leading-6 text-[#898c98]">Text-to-image and image-to-text retrieval are both part of the evaluation.</p></div>
        </div>
      </section>

      <footer className="bg-[#242943] px-5 py-8 text-white sm:px-8"><div className="mx-auto flex max-w-[1280px] flex-wrap items-center justify-between gap-4"><span className="text-sm font-black">vision<span className="text-[#d9ff70]">match</span></span><span className="text-[10px] text-white/45">An academic deep learning project · Flickr30k</span><Link href="/" className="flex items-center gap-2 text-xs font-bold text-[#d9ff70]">Back to search <ArrowUpRight size={14}/></Link></div></footer>
    </main>
  );
}
