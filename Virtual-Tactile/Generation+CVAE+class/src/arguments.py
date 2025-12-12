import argparse
from pathlib import Path
import os
import shutil
from datetime import datetime
import sys

def parse_argument():
    parser = argparse.ArgumentParser(description="Tactile Classification")
    
    if hasattr(sys, 'ps1') or 'spyder' in sys.argv[0]:
        args = parser.parse_args(args=[])
        args.name = f"spyder_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.epochs = 150
        args.data_dir = "E:\\Project\\IITP_AGI_MVTC\\Data"
        args.lr = 0.0001
        args.seed = 42
        args.batch_size = 32
        args.classnum = 5
        
    else:
        parser.add_argument("--name", type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        parser.add_argument("--epochs", type=int, default=1000)
        parser.add_argument("--data_dir", type=str, default="datadir/")
        parser.add_argument("--datasets", type=str, default="EEG")
        parser.add_argument("--try_n_times", type=int, default=1)
        parser.add_argument("--model_save", default=None, type=int)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument('--EEG_channels', type=int, default=30)
        parser.add_argument('--NIRS_channels', type=int, default=72)
        parser.add_argument('--output_size', type=int, default=128)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--ch_set", type=int, default=6)
        parser.add_argument("--model_type", type=str, default="early")
        args = parser.parse_args()
    
    args.data_dir = Path(args.data_dir)
    return args
