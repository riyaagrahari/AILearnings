import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { StatisticsResponse } from "../types";
import { Card } from "./Card";

const AXIS_COLOR = "#94a3b8"; // slate-400, readable in both themes

interface ChartCardProps {
  title: string;
  dataKey: "total_tokens" | "total_words" | "ratio";
  data: StatisticsResponse["languages"];
  color: string;
  valueFormatter?: (value: number) => string;
}

function ChartCard({ title, dataKey, data, color, valueFormatter }: ChartCardProps) {
  return (
    <Card title={title}>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-slate-200 dark:text-slate-800" />
            <XAxis dataKey="language" tick={{ fill: AXIS_COLOR, fontSize: 12 }} axisLine={{ stroke: AXIS_COLOR }} tickLine={false} />
            <YAxis tick={{ fill: AXIS_COLOR, fontSize: 12 }} axisLine={{ stroke: AXIS_COLOR }} tickLine={false} width={44} />
            <Tooltip
              formatter={(value: number) => (valueFormatter ? valueFormatter(value) : value)}
              contentStyle={{
                borderRadius: 10,
                border: "none",
                boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
                fontSize: 13,
              }}
            />
            <Bar dataKey={dataKey} fill={color} radius={[6, 6, 0, 0]} maxBarSize={56} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

export function StatsCharts({ stats }: { stats: StatisticsResponse }) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <ChartCard title="Total tokens" dataKey="total_tokens" data={stats.languages} color="#6366f1" />
      <ChartCard title="Total words" dataKey="total_words" data={stats.languages} color="#8b5cf6" />
      <ChartCard
        title="Xi = fertility (tokens / word)"
        dataKey="ratio"
        data={stats.languages}
        color="#ec4899"
        valueFormatter={(value) => value.toFixed(4)}
      />
    </div>
  );
}
