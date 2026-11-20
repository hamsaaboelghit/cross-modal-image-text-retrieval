"""Fair comparison of GRU, LSTM and Cross-Attention across three seeds."""
import argparse,csv,gc,json,time,traceback
from pathlib import Path
import tensorflow as tf
from src.training.experiment_runner import run_experiment
from src.training.experiment_utils import ROOT,atomic_json,evaluate_full_checkpoint

SEEDS=(42,123,456)
MODELS=({"name":"gru","encoder":"gru","cross":False},
        {"name":"lstm","encoder":"lstm","cross":False},
        {"name":"cross_attention","encoder":"lstm","cross":True})
FOLD,BATCH=1,4
LR,TEMP,OUT_DIM,DROPOUT=1e-4,0.07,512,0.1
TRAIN_STEPS,VAL_STEPS=200,200
OUT=ROOT/"results/fair_comparison_20261120"

def save_csv(rows):
    fields=["experiment","status","model","seed","fold","batch_size","train_steps",
        "validation_steps","best_validation_retrieval_loss","overall_validation_mean_recall",
        "overall_validation_mrr","training_time_minutes","checkpoint_path","error"]
    p=OUT/"comparison.csv"; t=OUT/"comparison.csv.tmp"
    with t.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows); f.flush()
    t.replace(p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list",action="store_true"); p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    print("FAIR COMPARISON | GRU / LSTM / Cross-Attention | 3 seeds")
    print(f"fold={FOLD}, batch={BATCH}, train={TRAIN_STEPS}, val={VAL_STEPS}")
    for s in SEEDS:
        for m in MODELS: print(f"seed={s}: {m['name']}")
    if a.list: print("List-only: no training started."); return
    OUT.mkdir(parents=True,exist_ok=True)
    meta={"seeds":list(SEEDS),"models":list(MODELS),"fold":FOLD,"batch_size":BATCH,
        "learning_rate":LR,"temperature":TEMP,"output_dim":OUT_DIM,"dropout":DROPOUT,
        "train_steps":TRAIN_STEPS,"validation_steps":VAL_STEPS,"epochs":1,
        "pretrained_resnet50":True,"backbone_frozen":True,"captions":5,
        "augmentation":"none","negative_sampling":"random","loss":"info_nce",
        "selection_split":"validation","test_set_used":False}
    mp,rp=OUT/"metadata.json",OUT/"results.json"
    if a.resume:
        if not mp.exists() or not rp.exists(): raise FileNotFoundError("Fair comparison resume files missing.")
        if json.loads(mp.read_text())!=meta: raise ValueError("Fair comparison settings differ.")
        rows=json.loads(rp.read_text())
    else:
        if mp.exists() or rp.exists(): raise FileExistsError("Fair outputs exist; use --resume.")
        rows=[]; atomic_json(mp,meta)
    completed={r["experiment"] for r in rows if r.get("status")=="completed"}
    idx,total=0,len(SEEDS)*len(MODELS)
    for seed in SEEDS:
        for m in MODELS:
            idx+=1; name=f"fair200_20261120_{m['name']}_fold{FOLD}_seed{seed}"
            if name in completed: print("[SKIP]",name); continue
            ckpt=ROOT/"checkpoints/experiments"/name
            if ckpt.exists():
                raise FileExistsError(f"Existing checkpoint directory: {ckpt}. Inspect/backup it first.")
            start=time.time(); print(f"\nFAIR RUN {idx}/{total}: {name}")
            try:
                h=run_experiment(experiment_name=name,model_type="full",encoder_type=m["encoder"],
                    use_cross_modal_attention=m["cross"],pretrained=True,unfreeze_last_block=False,
                    train_all_backbone=False,caption_count=5,image_augmentation="none",
                    negative_sampling_strategy="random",epochs=1,batch_size=BATCH,seed=seed,fold=FOLD,
                    learning_rate=LR,temperature=TEMP,loss_type="info_nce",output_dim=OUT_DIM,
                    dropout=DROPOUT,log_dir=f"runs/three_seed_200/seed_{seed}",
                    max_train_steps_per_epoch=TRAIN_STEPS,max_validation_steps=VAL_STEPS)
                met=evaluate_full_checkpoint(name,encoder_type=m["encoder"],cross_attention=m["cross"],batch_size=BATCH)
                losses=h.get("val_retrieval_loss") or h["val_loss"]
                rows.append({"experiment":name,"status":"completed","model":m["name"],"seed":seed,
                    "fold":FOLD,"batch_size":BATCH,"train_steps":TRAIN_STEPS,"validation_steps":VAL_STEPS,
                    "best_validation_retrieval_loss":min(losses),
                    "overall_validation_mean_recall":met["overall_validation_mean_recall"],
                    "overall_validation_mrr":met["overall_validation_mrr"],
                    "text_to_image":met["text_to_image"],"image_to_text":met["image_to_text"],
                    "training_time_minutes":(time.time()-start)/60,"checkpoint_path":met["checkpoint_path"],
                    "evaluation_split":"validation","test_set_used":False})
            except Exception as e:
                rows.append({"experiment":name,"status":"failed","model":m["name"],"seed":seed,
                    "error":str(e)[:1500],"training_time_minutes":(time.time()-start)/60})
                traceback.print_exc()
            finally:
                tf.keras.backend.clear_session(); gc.collect()
                atomic_json(rp,rows); save_csv(rows)
    print("Fair comparison stage finished; test set was not used.")

if __name__=="__main__": main()
