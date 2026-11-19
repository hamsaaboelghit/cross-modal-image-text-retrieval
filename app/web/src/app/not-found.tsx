import Link from "next/link";
import { ArrowLeft, BrainCircuit, Search } from "lucide-react";

export default function NotFound() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f7f7f4] px-6 text-[#171b2b]">
      <div className="w-full max-w-lg rounded-[30px] border border-[#e9e7df] bg-white p-9 text-center shadow-xl shadow-[#242943]/5 sm:p-12">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-[#242943] text-[#d9ff70]"><BrainCircuit size={25}/></div>
        <p className="mt-8 text-[10px] font-extrabold uppercase tracking-[2px] text-[#7770e8]">404 · Lost in the latent space</p>
        <h1 className="mt-3 text-4xl font-black tracking-[-2px]">Page not found<span className="text-[#d9ff70]">.</span></h1>
        <p className="mt-4 text-sm leading-7 text-[#858895]">This page isn&apos;t part of the collection. Let&apos;s get you back to the search workspace.</p>
        <Link href="/" className="mx-auto mt-8 inline-flex items-center gap-3 rounded-full bg-[#242943] px-6 py-3.5 text-xs font-extrabold text-white transition hover:bg-[#7770e8]"><ArrowLeft size={15}/> Back to home <Search size={14}/></Link>
      </div>
    </main>
  );
}
