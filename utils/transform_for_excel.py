import pandas as pd


def transform_for_excel(eval_df: pd.DataFrame, llm_name: str, batch_size: int) -> pd.DataFrame:
    dataset_names = ['ARTUR', 'COMMONVOICE', 'FLEURS', 'GVL', 'SOFES', 'VOXPOPULI']
    dataset_keys = ['artur', 'commonvoice', 'fleurs', 'gvl', 'sofes', 'voxpopuli']
    columns = [
        "LLM", "BEAM SIZE", "ALPHA", "BETA",
        *[f"{name} / WER" for name in dataset_names],
        "BATCH SIZE",
        "GPUS",
        *[f"{name} / DURATION" for name in dataset_names],
    ]

    dataset_map = dict(zip(dataset_names, dataset_keys))

    rows = []
    for beam_size in sorted(eval_df['beam_size'].unique()):
        row = {
            "LLM": llm_name,
            "BEAM SIZE": beam_size,
            "BATCH SIZE": batch_size,
        }
        subset = eval_df[eval_df['beam_size'] == beam_size]
        row["ALPHA"] = subset['alpha'].iloc[0]
        row["BETA"] = subset['beta'].iloc[0]

        for name, key in dataset_map.items():
            ds_row = subset[subset['results_file'].str.contains(key, case=False)]
            if not ds_row.empty:
                row[f"{name} / WER"] = ds_row['new_wer'].iloc[0]
                row[f"{name} / DURATION"] = ds_row['run_duration_in_seconds'].iloc[0]
            else:
                row[f"{name} / WER"] = None
                row[f"{name} / DURATION"] = None

        row["GPUS"] = subset['gpus'].iloc[0]
        rows.append(row)

    if all(row["ALPHA"] is None for row in rows):
        columns.remove("ALPHA")
        for row in rows:
            del row["ALPHA"]
    if all(row["BETA"] is None for row in rows):
        columns.remove("BETA")
        for row in rows:
            del row["BETA"]

    return pd.DataFrame(rows, columns=columns)