"use client";

import Link from "next/link";
import { ArrowLeft, ArrowUpRight, BarChart3, BrainCircuit, CheckCircle2, CircleDashed, Database, FlaskConical, Layers3, Sparkles } from "lucide-react";

const experiments = [
  { name: "Data reference", detail: "Five captions per image · reference setup", status: "Planned" },
  { name: "One caption", detail: "Single-caption training strategy", status: "Planned" },
  { name: "Light augmentation", detail: "Training-time image augmentation", status: "Planned" },
  { name: "Random triplet", detail: "Random negative sampling", status: "Planned" },
  { name: "Hardest triplet", detail: "Hard-negative sampling", status: "Planned" },
  { name: "Pretrained ResNet50", detail: "Fine-tuned pretrained visual encoder", status: "Planned" },
  { name: "ResNet50 from scratch", detail: "Visual encoder trained from initialization", status: "Planned" },
];

export default function ComparePage() {
  return (
    <main className="min-h-screen bg-[#f7f7f4] text-[#171b2b]">
      <header className="mx-auto flex max-w-[1280px] items-center justify-between px-5 py-6 sm:px-8 lg:px-12">
        <Link href="/" className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-[#242943] text-[#d9ff70]"><BrainCircuit size={21}/></span><span className="text-sm font-black tracking-tight">vision<span className="text-[#7770e8]">match</span></span></Link>
        <Link href="/" className="flex items-center gap-2 rounded-full border border-[#e6e4dc] bg-white px-4 py-2.5 text-xs font-bold text-[#6e7180] transition hover:border-[#7770e8] hover:text-[#7770e8]"><ArrowLeft size={14}/> Back to workspace</Link>
      </header>
      <div className="mx-auto max-w-[1280px] px-5 pb-20 pt-10 sm:px-8 lg:px-12">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div><p className="flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]"><Sparkles size={13}/> Experiment observatory</p><h1 className="mt-4 text-4xl font-black tracking-[-2px] sm:text-6xl">Model insights<span className="text-[#d9ed7c]">.</span></h1><p className="mt-4 max-w-[580px] text-sm leading-7 text-[#858895]">A transparent view of the Phase 5 experiment suite. Metrics will appear here after the training runs finish.</p></div>
          <span className="inline-flex items-center gap-2 rounded-full border border-[#eadfc9] bg-[#fffaf0] px-4 py-2.5 text-[10px] font-extrabold uppercase tracking-[1.2px] text-[#a47728]"><CircleDashed size={14}/> Awaiting experiment results</span>
        </div>
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[{label:"Experiments defined",value:"07",icon:FlaskConical},{label:"Dataset",value:"Flickr30k",icon:Database},{label:"Visual backbone",value:"ResNet50",icon:Layers3},{label:"Retrieval directions",value:"02",icon:BarChart3}].map((item) => <div key={item.label} className="rounded-[22px] border border-[#e8e6df] bg-white p-5"><span className="grid h-10 w-10 place-items-center rounded-xl bg-[#f0efff] text-[#7770e8]"><item.icon size={18}/></span><p className="mt-5 text-[10px] font-extrabold uppercase tracking-[1.4px] text-[#a0a1aa]">{item.label}</p><p className="mt-2 text-2xl font-black tracking-tight">{item.value}</p></div>)}
        </div>
        <section className="mt-8 rounded-[26px] border border-[#e8e6df] bg-white p-6 sm:p-8">
          <div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-[10px] font-extrabold uppercase tracking-[1.7px] text-[#7770e8]">Performance dashboard</p><h2 className="mt-2 text-xl font-black tracking-tight">Retrieval metrics</h2></div><span className="rounded-full bg-[#f2f1ed] px-3 py-1.5 text-[9px] font-bold uppercase tracking-[1px] text-[#8e9099]">No metrics loaded</span></div>
          <div className="mt-7 grid gap-4 md:grid-cols-2">
            {["Text → Image","Image → Text"].map((direction) => <div key={direction} className="rounded-2xl border border-dashed border-[#dedce9] bg-[#fbfbf9] p-6"><div className="flex items-center justify-between"><p className="text-sm font-extrabold">{direction}</p><BarChart3 size={17} className="text-[#7770e8]"/></div><div className="mt-8 grid grid-cols-3 gap-2">{["R@1","R@5","R@10"].map(metric => <div key={metric} className="rounded-xl bg-white p-3 text-center"><p className="text-[9px] font-bold text-[#9b9da6]">{metric}</p><p className="mt-2 text-xl font-black text-[#c0c0c7]">—</p></div>)}</div><p className="mt-4 text-[10px] text-[#a0a1aa]">Waiting for evaluation output</p></div>)}
          </div>
          <p className="mt-5 flex items-center gap-2 text-[10px] text-[#9698a2]"><CheckCircle2 size={13} className="text-[#8caa4a]"/> Metrics will be populated from the saved experiment evaluation files — no demo scores are displayed.</p>
        </section>
        <section className="mt-8 rounded-[26px] border border-[#e8e6df] bg-white p-6 sm:p-8">
          <div className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-[10px] font-extrabold uppercase tracking-[1.7px] text-[#7770e8]">Phase 05</p><h2 className="mt-2 text-xl font-black tracking-tight">Experiment matrix</h2></div><span className="text-[10px] font-semibold text-[#999ba5]">7 planned configurations</span></div>
          <div className="mt-6 divide-y divide-[#efeee9]">{experiments.map((experiment,index) => <div key={experiment.name} className="flex flex-wrap items-center gap-4 py-4"><span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-[#f2f1ed] text-[10px] font-black text-[#777b88]">{String(index+1).padStart(2,"0")}</span><div className="min-w-0 flex-1"><p className="text-sm font-bold">{experiment.name}</p><p className="mt-1 text-[11px] text-[#9698a2]">{experiment.detail}</p></div><span className="rounded-full bg-[#f5f3ee] px-3 py-1.5 text-[9px] font-extrabold uppercase tracking-[1px] text-[#999184]">{experiment.status}</span></div>)}</div>
        </section>
        <div className="mt-8 flex flex-wrap items-center justify-between gap-4 rounded-[24px] bg-[#242943] p-6 text-white sm:p-8"><div><p className="text-[10px] font-extrabold uppercase tracking-[1.6px] text-[#d9ff70]">Next step</p><h3 className="mt-2 text-xl font-black">Train first. Compare honestly.</h3><p className="mt-2 text-xs text-white/55">The dashboard is ready for real evaluation outputs.</p></div><Link href="/" className="inline-flex items-center gap-2 rounded-full bg-[#d9ff70] px-5 py-3 text-xs font-extrabold text-[#242943] transition hover:bg-white">Back to search <ArrowUpRight size={14}/></Link></div>
      </div>
    </main>
  );
}
