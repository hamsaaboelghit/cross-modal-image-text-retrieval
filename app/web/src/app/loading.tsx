import { BrainCircuit, Sparkles } from "lucide-react";

export default function Loading() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f7f7f4] px-6 text-[#171b2b]">
      <div className="w-full max-w-sm text-center">
        <div className="mx-auto grid h-16 w-16 place-items-center rounded-[22px] bg-[#242943] text-[#d9ff70] shadow-xl shadow-[#242943]/15"><BrainCircuit size={30}/></div>
        <p className="mt-6 text-lg font-black tracking-tight">vision<span className="text-[#7770e8]">match</span></p>
        <div className="mx-auto mt-8 h-1.5 w-48 overflow-hidden rounded-full bg-[#e8e6df]"><div className="h-full w-1/2 animate-pulse rounded-full bg-[#7770e8]"/></div>
        <p className="mt-4 flex items-center justify-center gap-2 text-[10px] font-bold uppercase tracking-[1.6px] text-[#9295a0]"><Sparkles size={12} className="text-[#7770e8]"/> Preparing your workspace</p>
      </div>
    </main>
  );
}
