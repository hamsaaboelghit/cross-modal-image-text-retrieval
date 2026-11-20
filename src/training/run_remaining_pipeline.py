"""Sequentially list or run remaining VisionMatch training stages."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STAGES=[
 ("Cross-Attention continuation +500 batches","src.training.run_cross_attention_resume",None,None),
 ("Fair comparison: GRU/LSTM/Cross-Attention x 3 seeds","src.training.run_three_models_seed","results/fair_comparison_20261120/metadata.json","results/fair_comparison_20261120/results.json"),
 ("GRU learning-rate study","src.training.run_hyperparameter_experiments","results/gru_hyperparameter_study/metadata.json","results/gru_hyperparameter_study/results.json"),
 ("GRU data strategy and pretrained-vs-scratch studies","src.training.run_phase5_experiments","results/gru_phase5_studies/phase5_run_metadata.json","results/gru_phase5_studies/phase5_results.json"),
]
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--list-only",action="store_true"); a=p.parse_args()
    print("Sequential pipeline; one TensorFlow process at a time.")
    for i,(label,module,_,_) in enumerate(STAGES,1): print(f"{i}. {label}: python -m {module}")
    if a.list_only: print("No training started."); return
    for label,module,meta,result in STAGES:
        cmd=[sys.executable,"-m",module]
        if meta and (ROOT/meta).exists() and (ROOT/result).exists(): cmd.append("--resume")
        print(f"\n{'='*70}\nSTART: {label}\n{'='*70}",flush=True)
        done=subprocess.run(cmd,check=False)
        if done.returncode: raise SystemExit(f"Stopped at {label} (exit={done.returncode}). Fix then resume that stage.")
    print("All stages completed.")
if __name__=="__main__": main()
