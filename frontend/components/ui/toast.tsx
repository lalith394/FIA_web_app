"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";

type Toast = {
  id: string;
  title?: string;
  description?: string;
  variant?: "success" | "error" | "info";
  duration?: number; // ms
};

type ToastContextValue = {
  show: (t: Omit<Toast, 'id'>) => void;
};

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const show = useCallback((t: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).slice(2, 9);
    setToasts((s) => [...s, { id, ...t }]);
  }, []);

  const remove = useCallback((id: string) => setToasts((s) => s.filter((t) => t.id !== id)), []);

  useEffect(() => {
    // Auto remove according to duration
    toasts.forEach((t) => {
      if (t.duration && t.duration > 0) {
        const timer = setTimeout(() => remove(t.id), t.duration);
        return () => clearTimeout(timer);
      }
      return undefined;
    });
  }, [toasts, remove]);

  return (
    <ToastContext.Provider value={{ show }}>
      {children}

      {/* Toast viewport (top-right) */}
      <div className="fixed top-6 right-6 z-50 flex flex-col gap-3 max-w-xs">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`p-3 rounded-md shadow-md border flex flex-col gap-1 text-sm bg-white dark:bg-slate-800 border-gray-200 dark:border-slate-700`}
          >
            {t.title && <div className="font-semibold">{t.title}</div>}
            {t.description && <div className="text-xs text-muted-foreground">{t.description}</div>}
            <div className="text-right pt-1">
              <button
                onClick={() => remove(t.id)}
                className="text-xs text-muted-foreground hover:underline"
              >
                Dismiss
              </button>
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside a ToastProvider");
  return ctx;
}

export default ToastProvider;
