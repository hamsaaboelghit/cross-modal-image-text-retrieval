"""GRU learning-rate study: 3 controlled trials, 200 train and 100 validation batches."""
import argparse, csv, gc, json, time, traceback
from pathlib import Path
import tensorflow as tf
from src.training.experiment_runner import run_experiment
from src.training.experiment_utils import ROOT, atomic_json, evaluate_full_checkpoint

SEED, FOLD, BATCH = 42, 1, 4
TRAIN_STEPS, VAL_STEPS = 200, 100
OUTPUT_DIM, DROPOUT, TEMPERATURE = 512, 0.1, 0.07
LRS = (5e-5, 1e-4, 2e-4)
OUT = ROOT / "results/gru_hyperparameter_study"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    configs = [{"name": f"gru_lr_{str(lr).replace('-', 'm').replace('.', '_')}",
                "learning_rate": lr} for lr in LRS]
    print("GRU HYPERPARAMETER STUDY | seed=42 | batch=4 | train=200 | val=100")
    for c in configs: print(c["name"], "LR=", c["learning_rate"])
    if a.list:
        print("List-only: no training started."); return
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"model":"GRU","seed":SEED,"fold":FOLD,"batch_size":BATCH,
        "train_steps":TRAIN_STEPS,"validation_steps":VAL_STEPS,"epochs":1,
        "learning_rates":list(LRS),"output_dim":OUTPUT_DIM,"dropout":DROPOUT,
        "temperature":TEMPERATURE,"selection_split":"validation","test_set_used":False}
    mp, rp = OUT/"metadata.json", OUT/"results.json"
    if a.resume:
        if not mp.exists() or not rp.exists(): raise FileNotFoundError("Resume metadata/results missing.")
        if json.loads(mp.read_text()) != meta: raise ValueError("Saved settings differ.")
        rows = json.loads(rp.read_text())
    else:
        if mp.exists() or rp.exists(): raise FileExistsError("Outputs exist; use --resume.")
        rows=[]; atomic_json(mp,meta)
    completed={r["experiment"] for r in rows if r.get("status")=="completed"}
    for i,c in enumerate(configs,1):
        name=c["name"]
        if name in completed: print("[SKIP]",name); continue
        start=time.time(); print(f"\nHYPERPARAMETER {i}/3: {name}")
        try:
            h=run_experiment(experiment_name=name,model_type="full",encoder_type="gru",
                use_cross_modal_attention=False,pretrained=True,unfreeze_last_block=False,
                train_all_backbone=False,caption_count=5,image_augmentation="none",
                negative_sampling_strategy="random",epochs=1,batch_size=BATCH,seed=SEED,
                fold=FOLD,learning_rate=c["learning_rate"],temperature=TEMPERATURE,
                loss_type="info_nce",output_dim=OUTPUT_DIM,dropout=DROPOUT,
                log_dir="runs/gru_hyperparameter_study",
                max_train_steps_per_epoch=TRAIN_STEPS,max_validation_steps=VAL_STEPS)
            m=evaluate_full_checkpoint(name,encoder_type="gru",batch_size=BATCH)
            losses=h.get("val_retrieval_loss") or h["val_loss"]
            rows.append({"experiment":name,"status":"completed","learning_rate":c["learning_rate"],
                "seed":SEED,"fold":FOLD,"batch_size":BATCH,"train_steps":TRAIN_STEPS,
                "validation_steps":VAL_STEPS,"best_validation_retrieval_loss":min(losses),
                "overall_validation_mean_recall":m["overall_validation_mean_recall"],
                "overall_validation_mrr":m["overall_validation_mrr"],
                "training_time_minutes":(time.time()-start)/60,
                "checkpoint_path":m["checkpoint_path"],"evaluation_split":"validation",
                "test_set_used":False})
        except Exception as e:
            rows.append({"experiment":name,"status":"failed","learning_rate":c["learning_rate"],
                         "error":str(e)[:1500]}); traceback.print_exc()
        finally:
            tf.keras.backend.clear_session(); gc.collect()
            atomic_json(rp,rows); save_csv(OUT,rows)
    ok=[r for r in rows if r.get("status")=="completed"]
    if ok:
        best=max(ok,key=lambda r:(r["overall_validation_mean_recall"],-r["best_validation_retrieval_loss"]))
        atomic_json(OUT/"best_configuration.json",{"best_experiment":best["experiment"],
            "learning_rate":best["learning_rate"],"selection_split":"validation","test_set_used":False})
        print("Best LR:",best["learning_rate"])
    atomic_json(rp,rows); save_csv(OUT,rows)

def save_csv(out,rows):
    fields=["experiment","status","learning_rate","seed","fold","batch_size","train_steps",
        "validation_steps","best_validation_retrieval_loss","overall_validation_mean_recall",
        "overall_validation_mrr","training_time_minutes","checkpoint_path","error"]
    path=out/"comparison.csv"; temp=out/"comparison.csv.tmp"
    with temp.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows); f.flush()
    temp.replace(path)

if __name__=="__main__": main()
