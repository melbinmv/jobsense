"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// How often to retry /health while the backend is waking up
const HEALTH_RETRY_MS = 3000;
// Per-attempt timeout for /health (a cold container can hang the request)
const HEALTH_TIMEOUT_MS = 15000;
// Max time to wait for a /query answer
const QUERY_TIMEOUT_MS = 90000;

type ApiStatus = "waking" | "ready" | "noindex";

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

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

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

function StatusBanner({
  status,
  waitedSecs,
}: {
  status: ApiStatus;
  waitedSecs: number;
}) {
  if (status === "ready") return null;

  if (status === "noindex") {
    return (
      <div className="shrink-0 px-6 py-2.5 bg-[#2a1a1a] border-b border-[#5c2a2a] text-xs text-[#f08b8b] text-center">
        The server is up, but the job index is empty. Build it with POST /index, then refresh.
      </div>
    );
  }

  return (
    <div className="shrink-0 px-6 py-2.5 bg-[#1a2236] border-b border-[#2a3a5c] flex items-center justify-center gap-2.5">
      <span className="w-3.5 h-3.5 rounded-full border-2 border-[#4dabf7] border-t-transparent animate-spin" />
      <span className="text-xs text-[#8b949e]">
        Waking up the server, this can take a minute…
        {waitedSecs >= 30 && " Still starting, almost there."}
      </span>
    </div>
  );
}

function EmptyState({
  onPick,
  disabled,
}: {
  onPick: (text: string) => void;
  disabled: boolean;
}) {
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
            onClick={() => onPick(s)}
            disabled={disabled}
            className="text-left px-4 py-2.5 rounded-xl bg-[#161b22] border border-[#2a3a5c] text-sm text-[#8b949e] hover:text-[#e6edf3] hover:border-[#4dabf7] transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:text-[#8b949e] disabled:hover:border-[#2a3a5c]"
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
  const [apiStatus, setApiStatus] = useState<ApiStatus>("waking");
  const [waitedSecs, setWaitedSecs] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isReady = apiStatus === "ready";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Wake-up check: poll /health until the API answers.
  // Re-runs whenever apiStatus goes back to "waking" (e.g. after a failed query).
  useEffect(() => {
    if (apiStatus !== "waking") return;

    let cancelled = false;
    const started = Date.now();
    const sleep = (ms: number) =>
      new Promise<void>((resolve) => setTimeout(resolve, ms));

    (async () => {
      while (!cancelled) {
        try {
          const res = await fetchWithTimeout(
            `${API_URL}/health`,
            { cache: "no-store" },
            HEALTH_TIMEOUT_MS
          );
          if (res.ok) {
            const data = await res.json();
            if (!cancelled) {
              setApiStatus(data.index_ready ? "ready" : "noindex");
            }
            return;
          }
        } catch {
          // Network error, timeout or non-JSON gateway page: the app is still
          // starting. Fall through and retry.
        }
        if (cancelled) return;
        setWaitedSecs(Math.round((Date.now() - started) / 1000));
        await sleep(HEALTH_RETRY_MS);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [apiStatus]);

  const handlePick = (text: string) => {
    setInput(text);
    textareaRef.current?.focus();
  };

  const handleSubmit = async () => {
    const query = input.trim();
    if (!query || isLoading || !isReady) return;

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
      const res = await fetchWithTimeout(
        `${API_URL}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        },
        QUERY_TIMEOUT_MS
      );

      if (!res.ok) {
        let detail = "Something went wrong";
        try {
          const err = await res.json();
          detail = err.detail || detail;
        } catch {
          // Non-JSON error body (e.g. a gateway error page)
          detail = `The server returned an error (${res.status}).`;
        }
        throw new Error(detail);
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
      let message: string;

      if (err instanceof DOMException && err.name === "AbortError") {
        message =
          "That took too long. The server may be waking up, so I'm reconnecting. Try again in a moment.";
        setWaitedSecs(0);
        setApiStatus("waking");
      } else if (err instanceof TypeError) {
        // fetch() itself failed: server unreachable or asleep
        message =
          "Lost connection to the server. It may have gone to sleep, so I'm reconnecting. Try again in a moment.";
        setWaitedSecs(0);
        setApiStatus("waking");
      } else if (err instanceof Error) {
        message = err.message;
      } else {
        message = "Failed to reach the JobSense API.";
      }

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

  const placeholder = isReady
    ? "Ask about jobs, salaries, or skills…"
    : apiStatus === "waking"
    ? "Waiting for the server to wake up…"
    : "The job index is empty";

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

      {/* Server status */}
      <StatusBanner status={apiStatus} waitedSecs={waitedSecs} />

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto">
          {messages.length === 0 ? (
            <EmptyState onPick={handlePick} disabled={!isReady} />
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
              placeholder={placeholder}
              disabled={!isReady}
              rows={1}
              className="flex-1 bg-transparent text-sm text-[#e6edf3] placeholder-[#8b949e] resize-none outline-none leading-relaxed disabled:cursor-not-allowed"
              style={{ maxHeight: "160px" }}
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading || !isReady}
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