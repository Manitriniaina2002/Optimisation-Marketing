import logging, sys, os
import pandas as pd
try:
    from src import config  # when run from project root
except Exception:
    # Fallback when run from within src/
    sys.path.append(os.path.dirname(__file__))
    import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('m5_marketing.log')])
logger = logging.getLogger(__name__)


def compute_kpis(df):
    df = df.copy()
    df['ctr'] = df['clicks'] / df['impressions'].replace(0, pd.NA)
    df['cvr'] = df['conversions'] / df['clicks'].replace(0, pd.NA)
    df['cpc'] = df['cost'] / df['clicks'].replace(0, pd.NA)
    df['cpa'] = df['cost'] / df['conversions'].replace(0, pd.NA)
    if 'revenue' in df.columns:
        df['roas'] = df['revenue'] / df['cost'].replace(0, pd.NA)
    return df


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, 'marketing_clean.csv')
    if not os.path.exists(path):
        logger.error('Missing marketing_clean.csv. Run M2 first.')
        return
    mkt = pd.read_csv(path)
    kpis = compute_kpis(mkt)
    # Save per-campaign KPIs
    kpis.to_csv(os.path.join(config.OUTPUT_DIR, 'campaign_kpis.csv'), index=False)

    # Aggregate overall KPIs
    summary = {
        'impressions': kpis['impressions'].sum(),
        'clicks': kpis['clicks'].sum(),
        'conversions': kpis['conversions'].sum() if 'conversions' in kpis.columns else pd.NA,
        'cost': kpis['cost'].sum(),
        'revenue': kpis['revenue'].sum() if 'revenue' in kpis.columns else pd.NA,
    }
    summary['ctr'] = (summary['clicks'] / summary['impressions']) if summary['impressions'] else pd.NA
    summary['cvr'] = (summary['conversions'] / summary['clicks']) if (summary['clicks'] and pd.notna(summary['conversions'])) else pd.NA
    summary['cpc'] = (summary['cost'] / summary['clicks']) if summary['clicks'] else pd.NA
    summary['cpa'] = (summary['cost'] / summary['conversions']) if (summary['conversions'] and summary['conversions'] != 0 and pd.notna(summary['conversions'])) else pd.NA
    summary['roas'] = (summary['revenue'] / summary['cost']) if ('revenue' in summary and summary['cost']) else pd.NA

    # Benchmark vs M1 targets
    targets = config.M1_TARGETS
    bench_rows = []
    def add_bench(name, value, target, comparator):
        if pd.isna(value) or target is None:
            status = 'N/A'
        else:
            if comparator == 'min':
                status = 'OK' if value >= target else 'GAP'
            elif comparator == 'max':
                status = 'OK' if value <= target else 'GAP'
            else:
                status = 'N/A'
        bench_rows.append({'kpi': name, 'value': value, 'target': target, 'status': status})

    add_bench('ctr', summary['ctr'], targets.get('ctr_min'), 'min')
    add_bench('cpc', summary['cpc'], targets.get('cpc_max'), 'max')
    add_bench('cpa', summary['cpa'], targets.get('cpa_max'), 'max')
    add_bench('roas', summary['roas'], targets.get('roi_min'), 'min')

    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(os.path.join(config.OUTPUT_DIR, 'kpi_benchmark.csv'), index=False)

    # Text summary
    txt_path = os.path.join(config.OUTPUT_DIR, 'kpi_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('KPI SUMMARY (Aggregated)\n')
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")
        f.write('\nBenchmark vs M1 Targets\n')
        for _, row in bench_df.iterrows():
            f.write(f"- {row['kpi']}: value={row['value']} | target={row['target']} | status={row['status']}\n")


if __name__ == '__main__':
    main()
