import { useEffect, useRef } from "react";
import * as echarts from "echarts";

interface EChartPanelProps {
  option: echarts.EChartsOption;
  className?: string;
}

export function EChartPanel({ option, className }: EChartPanelProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, undefined, { renderer: "canvas" });
    chart.setOption(option);
    const handleResize = () => chart.resize();
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.dispose();
    };
  }, [option]);

  return <div ref={ref} className={className ?? "h-80 w-full"} />;
}
