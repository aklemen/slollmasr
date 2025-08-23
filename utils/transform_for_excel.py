import pandas as pd


def transform_for_excel(eval_df: pd.DataFrame, llm_name: str, batch_size: int) -> pd.DataFrame:
    dataset_columns = [
        'ARTUR / TEST', 'COMMONVOICE / ALL', 'FLEURS / ALL',
        'GVL / ALL', 'SOFES / ALL', 'VOXPOPULI / ALL'
    ]

    excel_rows = []
    for _, row in eval_df.iterrows():
        excel_row = {
            'LLM': llm_name,
            'BEAM SIZE': row['beam_size'],
            'BATCH SIZE': batch_size,
            'RTFX': row['rtfx'],
            'DURATION': row['run_duration_in_seconds'],
            'GPUS': row['gpus'],
        }
        if row['alpha'] is not None:
            excel_row['ALPHA'] = row['alpha']
        if row['beta'] is not None:
            excel_row['BETA'] = row['beta']
        for col in dataset_columns:
            if col.split(' / ')[0].lower() in str(row['results_file']).lower():
                rounded_new_wer = round(row['new_wer'], 8)
                excel_row[col] = rounded_new_wer
            else:
                excel_row[col] = ''
        excel_rows.append(excel_row)

    columns = ['LLM', 'BEAM SIZE', 'ALPHA', 'BETA'] + dataset_columns + ['BATCH SIZE', 'DURATION', 'RTFX', 'GPUS']
    if 'ALPHA' not in excel_rows[0]:
        columns.remove('ALPHA')
    if 'BETA' not in excel_rows[0]:
        columns.remove('BETA')

    return pd.DataFrame(excel_rows, columns=columns)
