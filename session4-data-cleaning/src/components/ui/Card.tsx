import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  as?: "div" | "button" | "article";
  onClick?: () => void;
}

function cx(...parts: (string | false | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function Card({ children, className, hover = false, as = "div", onClick }: CardProps) {
  const Tag = as;
  return (
    <Tag
      onClick={onClick}
      className={cx(
        "glass rounded-2xl",
        hover && "glass-hover",
        as === "button" && "w-full text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-indigo/70",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export { cx };
