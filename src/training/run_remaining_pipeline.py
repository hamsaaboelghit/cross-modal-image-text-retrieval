"""Sequentially list or run the remaining VisionMatch training stages."""
import argparse,subprocess,sys
STAGES=[
 ("Cross-Attention continuation +500 batches","src.training.run_cross_attention_resume",[]),
 ("Fair comparison: GRU/LSTM/Cross-Attention x 3 seeds","src.training.run_three_models_seed",["--resume"]),
 ("GRU learning-rate study","src.training.run_hyperparameter_experiments",["--resume"]),
 ("GRU data strategy and pretrained-vs-scratch studies","src.training.run_phase5_experiments",["--resume"]),
]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-only",action="store_true")
    a=p.parse_args()
    print("Sequential pipeline; one TensorFlow process at a time.")
    for i,(label,module,_) in enumerate(STAGES,1): print(f"{i}. {label}: python -m {module}")
    if a.list_only: print("No training started."); return
    for label,module,resume_args in STAGES:
        print(f"\n{'='*70}\nSTART: {label}\n{'='*70}",flush=True)
        result=subprocess.run([sys.executable,"-m",module,*resume_args],check=False)
        if result.returncode:
            raise SystemExit(f"Stopped at {label} (exit={result.returncode}); resolve the error, then resume that stage.")
    print("All stages completed.")
if __name__=="__main__": main()
