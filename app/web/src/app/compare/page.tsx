"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowUpRight, BarChart3, BrainCircuit, Database, Layers3, RefreshCw, ShieldCheck, TriangleAlert } from "lucide-react";

type MetricSet = Record<string, number | undefined>;
type ModelMetric = {
  name?: string;
  experiment?: string;
  model_type?: string;
  encoder_type?: string | null;
  text_to_image?: MetricSet;
  image_to_text?: MetricSet;
  num_images?: number;
  embedding_dim?: number;
  latency_ms?: number;
  test_set_used?: boolean;
  source?: string;
};
type MetricsPayload = { available?: boolean; message?: string; source?: string; models?: ModelMetric[] };
type LoadState = "loading" | "ready" | "empty" | "error";
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const KS = ["recall@1", "recall@5", "recall@10"];

function MetricBars({ title, metrics }: { title: string; metrics?: MetricSet }) {
  const recorded = KS.some((key) => typeof metrics?.[key] === "number" && Number.isFinite(metrics[key]));
  return <div className="rounded-2xl border border-[#e9eaf1] bg-[#fafaff] p-5">
    <h3 className="text-xs font-extrabold">{title}</h3>
    <div className="mt-5 space-y-4">
      {KS.map((key) => {
        const value = metrics?.[key];
        const valid = typeof value === "number" && Number.isFinite(value);
        const percent = valid ? Math.max(0, Math.min(1, value)) * 100 : 0;
        return <div key={key}>
          <div className="mb-2 flex justify-between text-[10px] font-bold text-[#85899b]"><span>{key.toUpperCase()}</span><span className="tabular-nums text-[#55596b]">{valid ? `${(value * 100).toFixed(2)}%` : "Not recorded"}</span></div>
          <div className="h-2 overflow-hidden rounded-full bg-[#e9e9f1]"><div className="h-full rounded-full bg-[#7770e8] transition-all" style={{ width: `${percent}%` }} /></div>
        </div>;
      })}
    </div>
    {!recorded && <p className="mt-4 text-[10px] text-[#a0a2af]">No Recall@K values were included in this saved artifact.</p>}
  </div>;
}

export default function ComparePage() {
  const [data, setData] = useState<MetricsPayload | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState("");

  const loadMetrics = useCallback(async () => {
    setState("loading");
    setError("");
    try {
      const response = await fetch(`${API_BASE}/metrics`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Metrics endpoint returned HTTP ${response.status}.`);
      const payload = await response.json() as MetricsPayload;
      setData(payload);
      setState(payload.available && Array.isArray(payload.models) && payload.models.length > 0 ? "ready" : "empty");
    } catch (err) {
      setError(err instanceof Error ? err.message : "The metrics service could not be reached.");
      setState("error");
    }
  }, []);

  useEffect(() => { void loadMetrics(); }, [loadMetrics]);
  const models = Array.isArray(data?.models) ? data.models : [];

  return (
    <main className="min-h-screen bg-[#f6f7fb] text-[#171a2b]">
      <header className="mx-auto flex max-w-[1320px] items-center justify-between px-5 py-5 sm:px-8 lg:px-12">
        <Link href="/" className="flex items-center gap-3"><span className="grid h-10 w-10 place-items-center rounded-2xl bg-[#242943] text-[#d9ff70]"><BrainCircuit size={21} /></span><span className="text-base font-black">vision<span className="text-[#7770e8]">match</span></span></Link>
        <Link href="/" className="inline-flex items-center gap-2 rounded-full border border-[#e6e7ef] bg-white px-4 py-2.5 text-xs font-bold text-[#666a7b] transition hover:border-[#aaa3ef] hover:text-[#7770e8]"><ArrowLeft size={14} /> Back to workspace</Link>
      </header>

      <section className="mx-auto max-w-[1240px] px-5 pb-20 pt-12 sm:px-8 lg:px-12 lg:pt-16">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div><p className="flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]"><BarChart3 size={13} /> Evaluation dashboard</p><h1 className="mt-4 text-4xl font-black tracking-[-2.5px] sm:text-6xl">Model insights<span className="text-[#7770e8]">.</span></h1><p className="mt-4 max-w-2xl text-sm leading-7 text-[#85899b]">Compare the saved retrieval evaluation outputs. Values are loaded from the backend; the dashboard does not generate or fill in missing scores.</p></div>
          <button type="button" onClick={() => void loadMetrics()} disabled={state === "loading"} className="inline-flex items-center gap-2 rounded-full border border-[#e5e6ee] bg-white px-4 py-3 text-xs font-extrabold text-[#55596b] transition hover:border-[#aaa3ef] hover:text-[#7770e8] disabled:opacity-50"><RefreshCw size={14} className={state === "loading" ? "animate-spin" : ""} /> Refresh metrics</button>
        </div>

        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          {[{ label: "Dataset", value: "Flickr30k", icon: Database }, { label: "Visual backbone", value: "ResNet50", icon: Layers3 }, { label: "Retrieval directions", value: "Text ↔ Image", icon: BarChart3 }].map((item) => <div key={item.label} className="rounded-2xl border border-[#e8e9f0] bg-white p-5 shadow-[0_12px_35px_-30px_rgba(36,41,67,.3)]"><item.icon size={19} className="text-[#7770e8]" /><p className="mt-4 text-[9px] font-extrabold uppercase tracking-[1.7px] text-[#a0a2af]">{item.label}</p><p className="mt-1 text-lg font-black">{item.value}</p></div>)}
        </div>

        <section className="mt-8 rounded-[28px] border border-[#e8e9f0] bg-white p-5 shadow-[0_20px_65px_-48px_rgba(36,41,67,.25)] sm:p-8">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#eff0f5] pb-6"><div><p className="text-[10px] font-extrabold uppercase tracking-[1.8px] text-[#7770e8]">Saved evaluation artifacts</p><h2 className="mt-2 text-xl font-black">Recall@K comparison</h2></div><span className="rounded-full bg-[#f2f0ff] px-3 py-2 text-[10px] font-extrabold text-[#6e67d2]">{models.length} saved result(s)</span></div>

          {state === "loading" && <div className="flex items-center justify-center gap-3 py-16 text-xs font-semibold text-[#85899b]"><RefreshCw size={17} className="animate-spin text-[#7770e8]" /> Loading saved evaluation data…</div>}
          {state === "error" && <div role="alert" className="mt-6 flex items-start gap-3 rounded-2xl border border-[#f0dede] bg-[#fff8f8] p-5 text-sm text-[#a64d4d]"><TriangleAlert size={18} className="mt-0.5 shrink-0" /><div><p className="font-extrabold">Could not load evaluation data</p><p className="mt-1 text-xs leading-6">{error}</p><p className="mt-2 text-xs">Check that the FastAPI backend is running and that the configured metrics path is accessible.</p></div></div>}
          {state === "empty" && <div className="mt-7 rounded-2xl border border-dashed border-[#dcddea] bg-[#fafaff] px-6 py-12 text-center"><span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-[#f0efff] text-[#7770e8]"><BarChart3 size={22} /></span><h3 className="mt-4 text-sm font-extrabold">No saved evaluation results yet</h3><p className="mx-auto mt-2 max-w-lg text-xs leading-6 text-[#85899b]">{data?.message || "The API did not find a retrieval_metrics.json artifact in the configured evaluation directory."}</p><p className="mt-3 text-[10px] text-[#a0a2af]">This is an empty state, not a zero-score result. Run the evaluator and refresh this page when real metrics are saved.</p></div>}

          {state === "ready" && <div className="mt-6 space-y-5">{models.map((model, index) => <article key={model.experiment || model.name || index} className="rounded-2xl border border-[#e8e9f0] p-5 sm:p-6">
            <div className="flex flex-wrap items-start justify-between gap-4"><div><h3 className="text-lg font-black">{model.name || model.model_type || "Evaluated model"}</h3><p className="mt-1 text-xs text-[#85899b]">{model.experiment || model.encoder_type || "Experiment"} · {model.num_images ?? "—"} images · {model.embedding_dim ?? "—"}-D embeddings</p></div><span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-2 text-[9px] font-extrabold uppercase tracking-wider ${model.test_set_used === false ? "bg-[#f1f7e8] text-[#617b35]" : model.test_set_used === true ? "bg-[#fff2e9] text-[#a46b2e]" : "bg-[#f1f1f5] text-[#777b8c]"}`}>{model.test_set_used === false ? <><ShieldCheck size={12} /> Validation only</> : model.test_set_used === true ? "Test data included" : "Split not recorded"}</span></div>
            <div className="mt-5 grid gap-4 md:grid-cols-2"><MetricBars title="Text → Image" metrics={model.text_to_image} /><MetricBars title="Image → Text" metrics={model.image_to_text} /></div>
            <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-[#eff0f5] pt-4 text-[10px] text-[#a0a2af]"><span>Source: {model.source || data?.source || "saved evaluation artifact"}</span>{model.latency_ms != null && <span>Recorded latency: {model.latency_ms.toFixed(2)} ms</span>}</div>
          </article>)}</div>}

          <div className="mt-6 flex items-start gap-3 rounded-2xl bg-[#f7f7fb] p-4 text-[10px] leading-5 text-[#85899b]"><ShieldCheck size={15} className="mt-0.5 shrink-0 text-[#7770e8]" /><p>Only metrics present in saved evaluation artifacts are displayed. Missing values remain marked as “Not recorded”; validation metrics must not be presented as final test performance.</p></div>
        </section>
      </section>
      <footer className="mx-auto flex max-w-[1320px] items-center justify-between gap-4 px-5 py-8 text-[10px] text-[#999cad] sm:px-8 lg:px-12"><span>VisionMatch AI · Research evaluation</span><Link href="/about" className="inline-flex items-center gap-2 font-bold hover:text-[#7770e8]">Project methodology <ArrowUpRight size={12} /></Link></footer>
    </main>
  );
}
