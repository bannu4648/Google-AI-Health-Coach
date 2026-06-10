import type { ApiRow } from "../lib/api";

function formatCell(value: any): string {
  if (value == null) return "";
  if (typeof value === "object") return `${JSON.stringify(value).slice(0, 180)}...`;
  return String(value);
}

export function Table({ rows, columns }: { rows: ApiRow[]; columns: string[] }) {
  if (!rows.length) {
    return <p className="muted">No rows yet.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.id ?? index}>
              {columns.map((column) => (
                <td key={column}>{formatCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
