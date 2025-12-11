import json

with open("./data/stats_by_dataset.json") as f:
    data = json.load(f)

header = r"""
\begin{table}[h]
\centering
\caption{Accuracy metrics for models.}
\label{tab:accuracy_metrics}
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}
\hline
Name & \multicolumn{3}{c|}{AP, by J threshold, \%} & \multicolumn{2}{c|}{AP@[0.5:0.95],} & \multicolumn{3}{c|}{AR, by \# of} & \multicolumn{2}{c|}{AR@100, by} \\
  & \multicolumn{3}{c|}{} & \multicolumn{2}{c|}{by size, \%} & \multicolumn{3}{c|}{detections, \%} & \multicolumn{2}{c|}{size, \%} \\
\hline
& 0.5:0.95 & 0.5 & 0.75 & APm & APl & 1 & 10 & 100 & ARm & ARl \\
\hline"""
footer = r"""
\end{tabular}
\end{table}"""

row_template = "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n\\hline"

print(header)
for dataset in data:
    vals = dataset["stats"]
    print(
        row_template.format(
            dataset["name"],
            vals[0],
            vals[1],
            vals[2],
            vals[4],
            vals[5],
            vals[6],
            vals[7],
            vals[8],
            vals[10],
            vals[11],
        )
    )

print(footer)
