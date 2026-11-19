"use client";

import { useEffect } from "react";
import Link from "next/link";
import { AlertTriangle, ArrowLeft, RefreshCw } from "lucide-react";

export default function ErrorPage({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    // Keep the error available to the framework's error boundary without exposing details to visitors.
    console.error(error);
  }, [error]);

  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f7f7f4] px-6 text-[#171b2b]">
      <div className="w-full max-w-lg rounded-[30px] border border-[#e9e7df] bg-white p-9 text-center shadow-xl shadow-[#242943]/5 sm:p-12">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-[#fff4e8] text-[#bd7b35]"><AlertTriangle size={25}/></div>
        <p className="mt-8 text-[10px] font-extrabold uppercase tracking-[2px] text-[#bd7b35]">Something went off course</p>
        <h1 className="mt-3 text-4xl font-black tracking-[-2px]">Let&apos;s try that again.</h1>
        <p className="mt-4 text-sm leading-7 text-[#858895]">The workspace couldn't finish loading this view. You can retry or return to the homepage.</p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <button onClick={() => reset()} className="inline-flex items-center gap-2 rounded-full bg-[#242943] px-5 py-3.5 text-xs font-extrabold text-white transition hover:bg-[#7770e8]"><RefreshCw size={14}/> Try again</button>
          <Link href="/" className="inline-flex items-center gap-2 rounded-full border border-[#e6e4dc] bg-white px-5 py-3.5 text-xs font-extrabold text-[#55596b] transition hover:border-[#7770e8] hover:text-[#7770e8]"><ArrowLeft size={14}/> Home</Link>
        </div>
      </div>
    </main>
  );
}
