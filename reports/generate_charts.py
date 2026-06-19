import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv(
    "services/retrieval_service/reports/tables/metrics_report.csv"
)

metrics = ["MAP", "Recall@10", "P@10", "nDCG@10"]

for metric in metrics:
    plt.figure(figsize=(8, 5))

    plt.bar(df["Model"], df[metric])

    plt.title(f"Comparison of {metric}")
    plt.xlabel("Models")
    plt.ylabel(metric)

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(f"reports/figures/{metric}.png")

    plt.close()

print("Charts generated successfully.")