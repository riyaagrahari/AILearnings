interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="rounded-2xl border border-rose-200 bg-rose-50 p-5 text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-200">
      <div className="flex items-start gap-3">
        <svg className="mt-0.5 h-5 w-5 flex-none" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 6a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 6zm0 8a1 1 0 100-2 1 1 0 000 2z"
            clipRule="evenodd"
          />
        </svg>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">Something went wrong</p>
          <p className="mt-1 break-words text-sm opacity-90">{message}</p>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="mt-3 rounded-md bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-rose-700"
            >
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function LoadingSkeleton({ lines = 3, className = "" }: { lines?: number; className?: string }) {
  return (
    <div className={`animate-pulse space-y-3 ${className}`} role="status" aria-label="Loading">
      {Array.from({ length: lines }).map((_, index) => (
        <div
          key={index}
          className="h-4 rounded bg-slate-200 dark:bg-slate-800"
          style={{ width: `${85 - index * 12}%` }}
        />
      ))}
    </div>
  );
}

export function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M12 2a10 10 0 0110 10h-4a6 6 0 00-6-6V2z"
      />
    </svg>
  );
}
