import { Reveal } from "./Reveal";

interface SectionHeadingProps {
  index: string;
  eyebrow: string;
  title: string;
  description?: string;
  align?: "left" | "center";
}

export function SectionHeading({
  index,
  eyebrow,
  title,
  description,
  align = "left",
}: SectionHeadingProps) {
  const centered = align === "center";
  return (
    <Reveal className={centered ? "mx-auto max-w-3xl text-center" : "max-w-3xl"}>
      <div className={centered ? "flex items-center justify-center gap-3" : "flex items-center gap-3"}>
        <span className="font-mono text-xs text-accent-cyan/70">{index}</span>
        <span className="h-px w-8 bg-gradient-to-r from-accent-cyan/60 to-transparent" />
        <span className="eyebrow">{eyebrow}</span>
      </div>
      <h2 className="mt-4 text-3xl font-bold tracking-tight text-white sm:text-4xl md:text-5xl">
        {title}
      </h2>
      {description && (
        <p className="mt-5 text-base leading-relaxed text-slate-400 sm:text-lg">{description}</p>
      )}
    </Reveal>
  );
}
