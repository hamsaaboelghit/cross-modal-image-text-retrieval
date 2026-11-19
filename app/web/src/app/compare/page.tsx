"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, BarChart3, BrainCircuit, CircleDashed, Database, Layers3 } from "lucide-react";

type MetricsPayload = {
  available?: boolean;
  models?: Array<Record<string, unknown>>;
  [key: string]: unknown;
};
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export default function ComparePage() {
  const [data, setData] = useState<MetricsPayload | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/metrics`)
      .then(async response => {
        if (!response.ok) throw new Error("Could not load evaluation metrics.");
        return response.json();
      })
      .then(payload => { if (active) setData(payload); })
      .catch(err => { if (active) setError(err instanceof Error ? err.message : "Metrics API unavailable."); });
    return () => { active = false; };
  }, []);

  const models = Array.isArray(data?.models) ? data.models : [];
  return (
    <main className="min-h-screen bg-[#f7f7f4] text-[#171b2b]">
      <header className="mx-auto flex max-w-[1280px] items-center justify-between px-5 py-6 sm:px-8">
        <Link href="/" className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-xl bg-[#242943] text-[#d9ff70]"><BrainCircuit size={21}/></span><span className="text-sm font-black">vision<span className="text-[#7770e8]">match</span></span></Link>
        <Link href="/" className="inline-flex items-center gap-2 rounded-full border border-[#e6e4dc] bg-white px-4 py-2.5 text-xs font-bold"><ArrowLeft size={14}/> Back to search</Link>
      </header>
      <section className="mx-auto max-w-[1180px] px-5 pb-20 pt-10 sm:px-8">
        <p className="text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">Evaluation dashboard</p>
        <h1 className="mt-4 text-4xl font-black tracking-[-2px] sm:text-6xl">Model insights.</h1>
        <p className="mt-4 max-w-xl text-sm leading-7 text-[#858895]">This page reads saved evaluation artifacts from the backend. It does not invent metrics or treat planned experiments as completed results.</p>
        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          {[{label:"Dataset",value:"Flickr30k",icon:Database},{label:"Image encoder",value:"ResNet50",icon:Layers3},{label:"Retrieval directions",value:"2",icon:BarChart3}].map(item=><div key={item.label} className="rounded-2xl border border-[#e8e6df] bg-white p-5"><item.icon size={20} className="text-[#7770e8]"/><p className="mt-4 text-[10px] font-extrabold uppercase tracking-wider text-[#a0a1aa]">{item.label}</p><p className="mt-2 text-xl font-black">{item.value}</p></div>)}
        </div>
        <section className="mt-8 rounded-[26px] border border-[#e8e6df] bg-white p-6 sm:p-8">
          <div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-[10px] font-extrabold uppercase tracking-widest text-[#7770e8]">Saved evaluation output</p><h2 className="mt-2 text-xl font-black">Recall@K and model comparison</h2></div><span className="inline-flex items-center gap-2 rounded-full bg-[#f4f2ff] px-3 py-2 text-[10px] font-bold text-[#665fc5]"><CircleDashed size={13}/>{data?.available ? "Metrics loaded" : "Awaiting evaluation"}</span></div>
          {error && <p role="alert" className="mt-6 rounded-xl bg-red-50 p-4 text-sm text-red-700">{error}</p>}
          {!error && !data?.available && <div className="mt-6 rounded-2xl border border-dashed border-[#dedce9] bg-[#fbfbf9] p-8 text-center"><BarChart3 size={30} className="mx-auto text-[#7770e8]"/><p className="mt-4 font-bold">No final evaluation file found</p><p className="mx-auto mt-2 max-w-lg text-xs leading-6 text-[#858895]">{String(data?.message || "Run the final evaluation and save its summary JSON, then configure VISIONMATCH_METRICS_PATH to that file.")}</p></div>}
          {data?.available && models.length===0 && <pre className="mt-6 overflow-auto rounded-xl bg-[#242943] p-5 text-xs text-white">{JSON.stringify(data,null,2)}</pre>}
          {data?.available && models.length>0 && <div className="mt-6 overflow-x-auto"><table className="w-full min-w-[600px] text-left text-xs"><thead><tr className="border-b text-[#858895]"><th className="p-3">Model</th><th className="p-3">Text→Image</th><th className="p-3">Image→Text</th><th className="p-3">Latency</th></tr></thead><tbody>{models.map((model,index)=><tr key={String(model.model_type||model.name||index)} className="border-b border-[#efeee9]"><td className="p-3 font-bold">{String(model.name||model.model_type||"Model")}</td><td className="p-3">{String(model.text_to_image_recall||"—")}</td><td className="p-3">{String(model.image_to_text_recall||"—")}</td><td className="p-3">{String(model.latency_ms||"—")}</td></tr>)}</tbody></table></div>}
          {data?.available && <p className="mt-5 text-[10px] text-[#999ba5]">Source: {String(data.source || "configured evaluation summary")}</p>}
        </section>
      </section>
    </main>
  );
}
