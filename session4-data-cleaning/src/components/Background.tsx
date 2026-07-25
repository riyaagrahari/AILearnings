/** Static gradient-mesh + faint grid backdrop (GPU-cheap, no animation loop). */
export function Background() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-ink" />
      <div className="absolute inset-0 bg-grid-faint [background-size:56px_56px] opacity-40" />
      <div className="absolute -top-40 left-1/2 h-[46rem] w-[46rem] -translate-x-1/2 rounded-full bg-accent-indigo/20 blur-[140px]" />
      <div className="absolute top-1/3 -left-40 h-[34rem] w-[34rem] rounded-full bg-accent-cyan/10 blur-[130px]" />
      <div className="absolute bottom-0 right-0 h-[38rem] w-[38rem] rounded-full bg-accent-blue/10 blur-[140px]" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-ink" />
    </div>
  );
}
