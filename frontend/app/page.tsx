"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Source = {
  rank: number;
  score: number;
  title: string;
  company: string;
  location: string;
  salary: string;
  url: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  loading?: boolean;
};

function SourceCard({ source }: { source: Source }) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block p-3 rounded-lg bg-[#1a2236] hover:bg-[#1e2a42] transition-colors border border-[#2a3a5c]"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-[#e6edf3] truncate">{source.title}</p>
          <p className="text-xs text-[#8b949e] mt-0.5">{source.company} · {source.location}</p>
        </div>
        <span className="text-xs text-[#4dabf7] font-mono shrink-0">
          {(source.score * 100).toFixed(0)}%
        </span>
      </div>
      {source.salary && (
        <p className="text-xs text-[#3fb950] mt-1.5">{source.salary}</p>
      )}
    </a>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-[#8b949e] hover:text-[#4dabf7] transition-colors"
      >
        <span>{open ? "▾" : "▸"}</span>
        <span>{sources.length} source{sources.length !== 1 ? "s" : ""}</span>
      </button>
      {open && (
        <div className="mt-2 grid gap-2">
          {sources.map((s) => (
            <SourceCard key={s.rank} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-[#4dabf7] animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] px-4 py-2.5 rounded-2xl rounded-tr-sm bg-[#3b5bdb] text-[#e6edf3] text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%]">
        <div className="flex items-center gap-2 mb-1.5">
          <div className="w-5 h-5 rounded-full bg-[#4dabf7] flex items-center justify-center">
            <span className="text-[10px] font-bold text-[#0d1117]">J</span>
          </div>
          <span className="text-xs text-[#8b949e]">JobSense</span>
        </div>
        <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-[#161b22] border border-[#2a3a5c] text-[#e6edf3] text-sm leading-relaxed">
          {message.loading ? (
            <ThinkingDots />
          ) : (
            <>
            
              <div className="prose prose-invert prose-sm max-w-none
                prose-headings:text-[#e6edf3]
                prose-p:text-[#e6edf3]
                prose-strong:text-[#e6edf3]
                prose-li:text-[#e6edf3]
                prose-a:text-[#4dabf7]">
  <ReactMarkdown>{message.content}</ReactMarkdown>
</div>
              {message.sources && message.sources.length > 0 && (
                <Sources sources={message.sources} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  const suggestions = [
    "What data engineering roles are hiring in London?",
    "I know Python and Spark — what jobs suit me?",
    "What salary should I expect as an MLOps engineer?",
    "Which companies are hiring ML engineers right now?",
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-4">
      <div className="text-center">
        <div className="w-12 h-12 rounded-2xl bg-[#4dabf7] flex items-center justify-center mx-auto mb-4">
          <span className="text-2xl font-bold text-[#0d1117]">J</span>
        </div>
        <h2 className="text-xl font-semibold text-[#e6edf3]">JobSense</h2>
        <p className="text-sm text-[#8b949e] mt-1">
          Ask me anything about the job market
        </p>
      </div>
      <div className="grid gap-2 w-full max-w-md">
        {suggestions.map((s) => (
          <button
            key={s}
            onClick={() => {
              const input = document.getElementById("chat-input") as HTMLTextAreaElement;
              if (input) {
                input.value = s;
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.focus();
              }
            }}
            className="text-left px-4 py-2.5 rounded-xl bg-[#161b22] border border-[#2a3a5c] text-sm text-[#8b949e] hover:text-[#e6edf3] hover:border-[#4dabf7] transition-colors"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async () => {
    const query = input.trim();
    if (!query || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };

    const loadingMsg: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      loading: true,
    };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setInput("");
    setIsLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Something went wrong");
      }

      const data = await res.json();

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? {
                ...m,
                content: data.answer,
                sources: data.sources,
                loading: false,
              }
            : m
        )
      );
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to reach the JobSense API. Is it running on port 8000?";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? { ...m, content: message, loading: false }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex flex-col h-screen bg-[#0d1117]">
      {/* Header */}
      <header className="shrink-0 flex items-center justify-between px-6 py-4 border-b border-[#2a3a5c]">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-[#4dabf7] flex items-center justify-center">
            <span className="text-sm font-bold text-[#0d1117]">J</span>
          </div>
          <span className="font-semibold text-[#e6edf3]">JobSense</span>
        </div>
        <span className="text-xs text-[#8b949e]">
          Powered by Claude
        </span>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="flex flex-col gap-4">
              {messages.map((m) => (
                <ChatMessage key={m.id} message={m} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </main>

      {/* Input */}
      <div className="shrink-0 px-4 pb-6 pt-3 border-t border-[#2a3a5c]">
        <div className="max-w-2xl mx-auto">
          <div className="flex items-end gap-3 px-4 py-3 rounded-2xl bg-[#161b22] border border-[#2a3a5c] focus-within:border-[#4dabf7] transition-colors">
            <textarea
              id="chat-input"
              ref={textareaRef}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Ask about jobs, salaries, or skills…"
              rows={1}
              className="flex-1 bg-transparent text-sm text-[#e6edf3] placeholder-[#8b949e] resize-none outline-none leading-relaxed"
              style={{ maxHeight: "160px" }}
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              className="shrink-0 w-8 h-8 rounded-xl bg-[#4dabf7] flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[#74c0fc] transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path d="M5 12h14M12 5l7 7-7 7" stroke="#0d1117" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
          <p className="text-center text-xs text-[#8b949e] mt-2">
            Enter to send · Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}