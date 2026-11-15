
"use client";

import { useRef, useState } from "react";
import Image from "next/image";
import {
  ArrowRight,
  Image as ImageIcon,
  Search,
  Upload,
  Sparkles,
  Layers3,
  Clock3,
  Settings2,
  Menu,
  X,
  Zap,
  ImagePlus,
  FileImage,
  CheckCircle2,
  BrainCircuit,
} from "lucide-react";

type SearchMode = "text" | "image";

export default function Home() {
  const [mode, setMode] = useState<SearchMode>("text");
  const [query, setQuery] = useState("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [message, setMessage] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleModeChange = (newMode: SearchMode) => {
    setMode(newMode);
    setMessage("");
  };

  const handleImageUpload = (file?: File) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setMessage("Please select a valid image file.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setMessage("Image size must be less than 10 MB.");
      return;
    }

    const reader = new FileReader();

    reader.onload = () => {
      setSelectedImage(reader.result as string);
      setFileName(file.name);
      setMessage("");
    };

    reader.readAsDataURL(file);
  };

  const handleSearch = () => {
    if (mode === "text" && !query.trim()) {
      setMessage("Please enter a description before searching.");
      return;
    }

    if (mode === "image" && !selectedImage) {
      setMessage("Please upload an image before searching.");
      return;
    }

    setMessage(
      "Your input is ready! Connect the VisionMatch AI backend to start real retrieval."
    );
  };

  const clearImage = () => {
    setSelectedImage(null);
    setFileName("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f9fc] text-[#171923]">
      <div className="flex min-h-screen">
        {mobileMenuOpen && (
          <button
            aria-label="Close menu overlay"
            onClick={() => setMobileMenuOpen(false)}
            className="fixed inset-0 z-40 bg-black/30 lg:hidden"
          />
        )}

        <aside
          className={`fixed inset-y-0 left-0 z-50 flex w-[260px] flex-col border-r border-[#e9eaf1] bg-white px-5 py-6 transition-transform duration-300 lg:static lg:translate-x-0 ${
            mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="flex items-center justify-between px-1">
            <a href="#" className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[#7669f6] to-[#5145d8] text-white shadow-lg shadow-indigo-200">
                <BrainCircuit size={23} />
              </div>
              <div>
                <h1 className="text-[18px] font-bold tracking-tight">
                  VisionMatch
                </h1>
                <p className="text-[10px] font-semibold uppercase tracking-[2px] text-[#8b8fa2]">
                  AI Retrieval
                </p>
              </div>
            </a>

            <button
              onClick={() => setMobileMenuOpen(false)}
              className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 lg:hidden"
              aria-label="Close navigation"
            >
              <X size={19} />
            </button>
          </div>

          <div className="mt-11">
            <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[1.7px] text-[#a0a3b1]">
              Workspace
            </p>

            <nav className="space-y-1">
              <a
                href="#home"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl bg-[#f0efff] px-3 py-3 text-sm font-semibold text-[#6258e8]"
              >
                <Layers3 size={18} />
                Overview
              </a>

              <button
                onClick={() => {
                  handleModeChange("text");
                  setMobileMenuOpen(false);
                }}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm font-medium transition ${
                  mode === "text"
                    ? "bg-[#f0efff] text-[#6258e8]"
                    : "text-[#777b8c] hover:bg-gray-50"
                }`}
              >
                <Search size={18} />
                Text to Image
              </button>

              <button
                onClick={() => {
                  handleModeChange("image");
                  setMobileMenuOpen(false);
                }}
                className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm font-medium transition ${
                  mode === "image"
                    ? "bg-[#f0efff] text-[#6258e8]"
                    : "text-[#777b8c] hover:bg-gray-50"
                }`}
              >
                <ImageIcon size={18} />
                Image to Text
              </button>
            </nav>
          </div>

          <div className="mt-9">
            <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[1.7px] text-[#a0a3b1]">
              General
            </p>

            <nav className="space-y-1">
              <a
                href="#history"
                className="flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-[#777b8c] transition hover:bg-gray-50"
              >
                <Clock3 size={18} />
                Search History
              </a>

              <a
                href="#about"
                className="flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-[#777b8c] transition hover:bg-gray-50"
              >
                <Settings2 size={18} />
                About Project
              </a>
            </nav>
          </div>

          <div className="mt-auto">
            <div className="rounded-2xl border border-[#e9e7ff] bg-gradient-to-br from-[#f7f6ff] to-[#eeecff] p-4">
              <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-xl bg-white text-[#675be8] shadow-sm">
                <Sparkles size={19} />
              </div>
              <h3 className="text-sm font-bold">Powered by Deep Learning</h3>
              <p className="mt-2 text-xs leading-5 text-[#777b8c]">
                Discover semantic connections between images and natural
                language.
              </p>
              <div className="mt-4 flex items-center gap-2 text-[10px] font-semibold text-[#6258e8]">
                <span className="h-1.5 w-1.5 rounded-full bg-[#6258e8]" />
                VisionMatch AI
              </div>
            </div>

            <p className="mt-5 text-center text-[10px] text-[#a0a3b1]">
              VisionMatch AI · Academic Project
            </p>
          </div>
        </aside>

        <main id="home" className="min-w-0 flex-1">
          <header className="sticky top-0 z-30 flex h-[76px] items-center justify-between border-b border-[#e9eaf1] bg-white/90 px-5 backdrop-blur-xl sm:px-8 lg:px-10">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="rounded-lg p-2 text-[#62677a] hover:bg-gray-100 lg:hidden"
                aria-label="Open navigation"
              >
                <Menu size={21} />
              </button>
              <div>
                <p className="text-xs text-[#9699a8]">Workspace / Overview</p>
                <h2 className="mt-0.5 text-sm font-bold sm:text-base">
                  AI Retrieval Studio
                </h2>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden items-center gap-2 rounded-full border border-[#f1dcae] bg-[#fffaf0] px-3 py-2 sm:flex">
                <span className="h-2 w-2 rounded-full bg-[#e6a43c]" />
                <span className="text-[11px] font-semibold text-[#a47728]">
                  Backend not connected
                </span>
              </div>

              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#eeeaff] text-sm font-bold text-[#6258e8]">
                VM
              </div>
            </div>
          </header>

          <div className="mx-auto max-w-[1440px] px-5 py-9 sm:px-8 lg:px-10 lg:py-11">
            <section className="relative overflow-hidden rounded-[26px] bg-gradient-to-br from-[#26234f] via-[#393276] to-[#6559df] px-6 py-9 text-white shadow-xl shadow-indigo-100 sm:px-10 sm:py-11">
              <div className="pointer-events-none absolute -right-20 -top-36 h-80 w-80 rounded-full border border-white/10" />
              <div className="pointer-events-none absolute -right-5 -top-20 h-60 w-60 rounded-full border border-white/10" />
              <div className="pointer-events-none absolute -bottom-32 right-[28%] h-72 w-72 rounded-full bg-[#a49aff]/10 blur-3xl" />

              <div className="relative z-10 max-w-[650px]">
                <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3 py-1.5 text-[10px] font-semibold tracking-wide text-[#e5e1ff]">
                  <Sparkles size={13} />
                  CROSS-MODAL AI TECHNOLOGY
                </div>

                <h1 className="text-3xl font-bold leading-tight tracking-tight sm:text-[42px]">
                  Search beyond
                  <br className="hidden sm:block" /> words and images.
                </h1>

                <p className="mt-4 max-w-[520px] text-sm leading-7 text-[#d5d1f5] sm:text-[15px]">
                  Explore the connection between visual content and natural
                  language. Find relevant images from text, or discover
                  descriptions from images.
                </p>

                <div className="mt-7 flex flex-wrap gap-3">
                  <div className="flex items-center gap-2 rounded-lg border border-white/15 bg-white/10 px-3 py-2 text-xs text-white/90">
                    <Zap size={14} />
                    Deep Learning
                  </div>
                  <div className="flex items-center gap-2 rounded-lg border border-white/15 bg-white/10 px-3 py-2 text-xs text-white/90">
                    <ImageIcon size={14} />
                    Image Retrieval
                  </div>
                  <div className="flex items-center gap-2 rounded-lg border border-white/15 bg-white/10 px-3 py-2 text-xs text-white/90">
                    <Search size={14} />
                    Text Retrieval
                  </div>
                </div>
              </div>

              <div className="pointer-events-none absolute bottom-0 right-10 hidden h-full w-[310px] items-center justify-center lg:flex">
                <div className="relative flex h-[220px] w-[220px] items-center justify-center rounded-full border border-white/15 bg-white/[0.04]">
                  <div className="absolute h-[165px] w-[165px] rounded-full border border-white/20" />
                  <div className="absolute h-[110px] w-[110px] rounded-full border border-white/20" />
                  <div className="flex h-[76px] w-[76px] items-center justify-center rounded-2xl border border-white/30 bg-white/15 shadow-2xl backdrop-blur-xl">
                    <BrainCircuit size={38} />
                  </div>
                  <div className="absolute left-2 top-7 flex h-11 w-11 items-center justify-center rounded-xl border border-white/20 bg-white/15 backdrop-blur-md">
                    <ImageIcon size={21} />
                  </div>
                  <div className="absolute bottom-6 right-2 flex h-11 w-11 items-center justify-center rounded-xl border border-white/20 bg-white/15 backdrop-blur-md">
                    <Search size={21} />
                  </div>
                  <div className="absolute right-0 top-5 h-3 w-3 rounded-full bg-[#b9afff] shadow-[0_0_20px_#b9afff]" />
                  <div className="absolute bottom-5 left-1 h-2 w-2 rounded-full bg-[#b9afff]" />
                </div>
              </div>
            </section>

            <section className="mt-10">
              <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-[1.8px] text-[#8c8fa1]">
                    Get started
                  </p>
                  <h2 className="mt-2 text-xl font-bold tracking-tight sm:text-2xl">
                    What are you looking for?
                  </h2>
                  <p className="mt-2 text-sm text-[#85899a]">
                    Choose how you want to search across the image-text space.
                  </p>
                </div>
                <div className="hidden items-center gap-2 text-xs text-[#85899a] sm:flex">
                  <CheckCircle2 size={15} className="text-[#7e75e8]" />
                  Two-way semantic retrieval
                </div>
              </div>

              <div className="overflow-hidden rounded-2xl border border-[#e8eaf1] bg-white shadow-sm">
                <div className="flex flex-wrap gap-2 border-b border-[#eff0f5] p-4 sm:px-6">
                  <button
                    onClick={() => handleModeChange("text")}
                    className={`flex items-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition ${
                      mode === "text"
                        ? "bg-[#6258e8] text-white shadow-md shadow-indigo-100"
                        : "text-[#777b8c] hover:bg-[#f5f5fa]"
                    }`}
                  >
                    <Search size={17} />
                    Text to Image
                  </button>

                  <button
                    onClick={() => handleModeChange("image")}
                    className={`flex items-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition ${
                      mode === "image"
                        ? "bg-[#6258e8] text-white shadow-md shadow-indigo-100"
                        : "text-[#777b8c] hover:bg-[#f5f5fa]"
                    }`}
                  >
                    <ImageIcon size={17} />
                    Image to Text
                  </button>
                </div>

                <div className="p-5 sm:p-7">
                  {mode === "text" ? (
                    <div>
                      <label
                        htmlFor="search-query"
                        className="mb-3 block text-sm font-semibold text-[#343748]"
                      >
                        Describe the image you want to find
                      </label>

                      <textarea
                        id="search-query"
                        value={query}
                        onChange={(e) => {
                          setQuery(e.target.value);
                          setMessage("");
                        }}
                        placeholder="Example: A golden retriever playing with a ball in a green park..."
                        rows={4}
                        className="w-full resize-none rounded-xl border border-[#e5e7ef] bg-[#fbfbfd] px-4 py-4 text-sm leading-6 text-[#25283a] outline-none transition placeholder:text-[#a6a9b6] focus:border-[#8279ee] focus:ring-4 focus:ring-[#6258e8]/10"
                      />

                      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                        <p className="text-[11px] text-[#9a9dad]">
                          Use natural language to describe the visual content.
                        </p>
                        <p className="text-[11px] tabular-nums text-[#9a9dad]">
                          {query.length} characters
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <label className="mb-3 block text-sm font-semibold text-[#343748]">
                        Upload an image to find matching descriptions
                      </label>

                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(e) =>
                          handleImageUpload(e.target.files?.[0])
                        }
                      />

                      {selectedImage ? (
                        <div className="flex flex-col gap-4 rounded-xl border border-[#e5e7ef] bg-[#fbfbfd] p-4 sm:flex-row sm:items-center">
                          <Image
                            src={selectedImage}
                            alt="Selected upload preview"
                            width={192}
                            height={144}
                            unoptimized
                            className="h-36 w-full rounded-lg object-contain sm:w-48"
                          />
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2 text-sm font-semibold text-[#343748]">
                              <FileImage size={17} className="text-[#6258e8]" />
                              Image selected
                            </div>
                            <p className="mt-2 break-all text-xs text-[#85899a]">
                              {fileName}
                            </p>
                            <button
                              onClick={clearImage}
                              className="mt-4 flex items-center gap-2 text-xs font-semibold text-red-500 hover:text-red-600"
                            >
                              <X size={14} />
                              Remove image
                            </button>
                          </div>
                        </div>
                      ) : (
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          onDragOver={(e) => e.preventDefault()}
                          onDrop={(e) => {
                            e.preventDefault();
                            handleImageUpload(e.dataTransfer.files?.[0]);
                          }}
                          className="flex min-h-[190px] w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-[#dfe0ed] bg-[#fbfbfd] px-4 py-8 text-center transition hover:border-[#8279ee] hover:bg-[#f8f7ff]"
                        >
                          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#efedff] text-[#6258e8]">
                            <Upload size={22} />
                          </div>
                          <p className="mt-4 text-sm font-semibold text-[#343748]">
                            Click to upload or drag and drop
                          </p>
                          <p className="mt-2 text-xs text-[#9699a8]">
                            PNG, JPG, JPEG or WEBP · Maximum 10 MB
                          </p>
                        </button>
                      )}
                    </div>
                  )}

                  {message && (
                    <div className="mt-4 rounded-xl border border-[#e8e5ff] bg-[#f7f6ff] px-4 py-3 text-xs leading-5 text-[#5c53c7]">
                      {message}
                    </div>
                  )}

                  <div className="mt-6 flex flex-wrap items-center justify-between gap-4 border-t border-[#eff0f5] pt-5">
                    <div className="flex items-center gap-2 text-[11px] text-[#999cac]">
                      <Sparkles size={14} className="text-[#8279ee]" />
                      Semantic matching powered by deep learning
                    </div>

                    <button
                      onClick={handleSearch}
                      className="flex items-center justify-center gap-2 rounded-xl bg-[#6258e8] px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-100 transition hover:-translate-y-0.5 hover:bg-[#5146d8] hover:shadow-xl"
                    >
                      <Search size={16} />
                      Search
                      <ArrowRight size={16} />
                    </button>
                  </div>
                </div>
              </div>
            </section>

            <section className="mt-9" id="history">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-bold tracking-tight">
                    Search Results
                  </h2>
                  <p className="mt-1 text-xs text-[#9295a5]">
                    Your retrieved matches will appear here.
                  </p>
                </div>
                <div className="rounded-lg border border-[#e8eaf1] bg-white px-3 py-2 text-[11px] font-medium text-[#85899a]">
                  Waiting for search
                </div>
              </div>

              <div className="flex min-h-[230px] flex-col items-center justify-center rounded-2xl border border-dashed border-[#dfe1eb] bg-white px-5 py-9 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#f0efff] text-[#7066e8]">
                  <ImagePlus size={25} />
                </div>
                <h3 className="mt-4 text-sm font-bold text-[#343748]">
                  Your results will show up here
                </h3>
                <p className="mt-2 max-w-[390px] text-xs leading-6 text-[#9295a5]">
                  Start by entering a text description or uploading an image.
                  Once the backend is connected, matching images and captions
                  will be displayed in this area.
                </p>
                <div className="mt-5 flex items-center gap-2 rounded-lg bg-[#f7f7fb] px-3 py-2 text-[10px] font-medium text-[#8b8ea0]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#e6a43c]" />
                  Retrieval engine is not connected yet
                </div>
              </div>
            </section>

            <section
              id="about"
              className="mt-9 grid gap-4 pb-8 md:grid-cols-3"
            >
              <div className="rounded-2xl border border-[#e8eaf1] bg-white p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#f0efff] text-[#6258e8]">
                  <Search size={19} />
                </div>
                <h3 className="mt-4 text-sm font-bold">Text to Image</h3>
                <p className="mt-2 text-xs leading-6 text-[#85899a]">
                  Search a visual dataset using natural language descriptions.
                </p>
              </div>

              <div className="rounded-2xl border border-[#e8eaf1] bg-white p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#edf7ff] text-[#3e8bc7]">
                  <ImageIcon size={19} />
                </div>
                <h3 className="mt-4 text-sm font-bold">Image to Text</h3>
                <p className="mt-2 text-xs leading-6 text-[#85899a]">
                  Retrieve relevant textual descriptions from a visual query.
                </p>
              </div>

              <div className="rounded-2xl border border-[#e8eaf1] bg-white p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#fff4e8] text-[#d18a35]">
                  <BrainCircuit size={19} />
                </div>
                <h3 className="mt-4 text-sm font-bold">Deep Learning</h3>
                <p className="mt-2 text-xs leading-6 text-[#85899a]">
                  Shared semantic embeddings connect visual and textual
                  representations.
                </p>
              </div>
            </section>

            <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-[#e8eaf1] py-6 text-[10px] text-[#a0a3b1]">
              <p>© 2026 VisionMatch AI. Academic Deep Learning Project.</p>
              <div className="flex items-center gap-2">
                <span>Built with</span>
                <span className="font-semibold text-[#777b8c]">Next.js</span>
                <span>·</span>
                <span className="font-semibold text-[#777b8c]">PyTorch</span>
              </div>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}
