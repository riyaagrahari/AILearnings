import { useCallback, useState } from "react";

/** Copies `text` to the clipboard and reports "copied" for `resetAfterMs`. */
export function useClipboard(resetAfterMs = 1500): [boolean, (text: string) => void] {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(
    (text: string) => {
      navigator.clipboard
        .writeText(text)
        .then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), resetAfterMs);
        })
        .catch(() => {
          // Clipboard API can be unavailable (permissions/insecure context) --
          // fail silently rather than crash the UI over a copy button.
        });
    },
    [resetAfterMs],
  );

  return [copied, copy];
}
