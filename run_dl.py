import argparse
import os
from dl_pipeline import DLPipeline

def main():
    parser = argparse.ArgumentParser(description="Esegui la Pipeline di Deep Learning (LSTM)")
    parser.add_argument("--train", required=True, help="Path al file train_timeseries.csv")
    parser.add_argument("--test", required=True, help="Path al file test_timeseries.csv")
    parser.add_argument("--outdir", required=True, help="Cartella per output (CSV, plots)")
    
    parser.add_argument("--lags", type=int, nargs='+', required=True, help="Lista dei lag (es. --lags 7 14)")
    parser.add_argument("--macro", action="store_true", help="Usa aggregazione per macro-aree")
    parser.add_argument("--clip", action="store_true", help="Applica clipping al 95° percentile per gli outlier")

    # Parametri Deep Learning (Default inclusi)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pipeline = DLPipeline(args.train, args.test, args.outdir, args.macro)
    
    pipeline.run(
        lags=args.lags,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        apply_clipping=args.clip
    )

if __name__ == "__main__":
    main()