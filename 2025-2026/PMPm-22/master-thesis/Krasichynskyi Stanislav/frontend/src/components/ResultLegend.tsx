type ResultLegendProps = {
  min: number;
  max: number;
  title: string;
};

export function ResultLegend({ min, max, title }: ResultLegendProps) {
  return (
    <div className="panel">
      <div className="panel-title">{title}</div>

      <div
        style={{
          height: 18,
          borderRadius: 999,
          background: "linear-gradient(90deg, rgb(0,0,255), cyan, yellow, rgb(255,0,0))",
          border: "1px solid rgba(255,255,255,0.1)",
        }}
      />

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 8,
          color: "#cbd5e1",
          fontSize: 13,
        }}
      >
        <span>{min.toFixed(4)}</span>
        <span>{max.toFixed(4)}</span>
      </div>
    </div>
  );
}
