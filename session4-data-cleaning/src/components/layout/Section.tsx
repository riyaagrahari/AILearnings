import type { ReactNode } from "react";

interface SectionProps {
  id: string;
  children: ReactNode;
  className?: string;
}

export function Section({ id, children, className }: SectionProps) {
  return (
    <section
      id={id}
      className={`relative mx-auto max-w-6xl scroll-mt-20 px-6 py-24 ${className ?? ""}`}
    >
      {children}
    </section>
  );
}
