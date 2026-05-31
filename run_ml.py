import argparse
import os
from src.ml_pipeline import MLPipeline
from src.config_unitelma import CLASSIFIERS

def main():
    parser = argparse.ArgumentParser(description="Esegui la Pipeline di Machine Learning")
    parser.add_argument("--train", required=True, help="Path al file train_timeseries.csv")
    parser.add_argument("--test", required=True, help="Path al file test_timeseries.csv")
    parser.add_argument("--outdir", required=True, help="Cartella per output (CSV, plots)")
    
    # Parametri Pipeline
    parser.add_argument("--lags", type=int, nargs='+', required=True, help="Lista dei lag (es. --lags 7 14)")
    parser.add_argument("--macro", action="store_true", help="Usa aggregazione per macro-aree")
    parser.add_argument("--fs", action="store_true", help="Applica Feature Selection (Solo se non usi macro)")
    parser.add_argument("--agg", type=str, choices=['flatten', 'sum'], default='flatten', help="Time aggregation")
    parser.add_argument("--clip", action="store_true", help="Applica clipping al 95° percentile per gli outlier")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pipeline = MLPipeline(args.train, args.test, args.outdir, args.macro)
    
    pipeline.run(
        lags=args.lags,
        classifiers=CLASSIFIERS,
        apply_clipping=args.clip,
        use_fs=args.fs,
        time_aggregation=args.agg
    )

if __name__ == "__main__":
    main()