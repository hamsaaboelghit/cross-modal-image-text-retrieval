"""GRU data-strategy and ResNet50 initialization studies with a fixed compute budget."""
import argparse, csv, gc, json, time, traceback
from pathlib import Path
import tensorflow as tf
from src.training.experiment_runner import run_experiment
from src.training.experiment_utils import ROOT, atomic_json, evaluate_full_checkpoint

SEED,FOLD=42,1
LR,BATCH,OUTPUT_DIM,DROPOUT,TEMP=1e-4,1,512,0.1,0.07
TRAIN_STEPS,VAL_STEPS=200,100
EVAL_BATCH_SIZE=4
OUT=ROOT/"results/gru_phase5_studies"

def build_experiments():
    common={"encoder_type":"gru","use_cross_modal_attention":False,"caption_count":5,
        "image_augmentation":"none","negative_sampling_strategy":"random","pretrained":True,
        "unfreeze_last_block":False,"train_all_backbone":False}
    specs=[
        ("gru_phase5_data_reference",{}),
        ("gru_phase5_one_caption",{"caption_count":1}),
        ("gru_phase5_light_augmentation",{"image_augmentation":"light"}),
        # Both init conditions fine-tune the same final ResNet block; BatchNorm stays frozen.
        ("gru_phase5_resnet50_pretrained",{"unfreeze_last_block":True}),
        ("gru_phase5_resnet50_scratch",{"pretrained":False,"unfreeze_last_block":True}),
    ]
    return [{"name":n,**common,**v} for n,v in specs]

def save(out,rows):
    atomic_json(out/"phase5_results.json",rows)
    fields=["experiment","status","seed","batch_size","train_steps","validation_steps",
        "caption_count","image_augmentation","pretrained","unfreeze_last_block",
        "best_validation_retrieval_loss","overall_validation_mean_recall",
        "overall_validation_mrr","training_time_minutes","checkpoint_path","error"]
    p=out/"phase5_comparison.csv"; t=out/"phase5_comparison.csv.tmp"
    with t.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows); f.flush()
    t.replace(p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list",action="store_true"); p.add_argument("--resume",action="store_true")
    a=p.parse_args(); configs=build_experiments()
    print("PHASE 5 | GRU | train_batch=1 | eval_batch=4 | train=200 | val=100")
    for c in configs: print(c["name"],{k:c[k] for k in ("caption_count","image_augmentation","pretrained","unfreeze_last_block")})
    if a.list: print("List-only: no training started."); return
    OUT.mkdir(parents=True,exist_ok=True)
    meta={"experiments":configs,"seed":SEED,"fold":FOLD,"learning_rate":LR,"batch_size":BATCH,
        "train_steps":TRAIN_STEPS,"validation_steps":VAL_STEPS,"epochs":1,"output_dim":OUTPUT_DIM,
        "dropout":DROPOUT,"temperature":TEMP,"selection_split":"validation","test_set_used":False}
    mp,rp=OUT/"phase5_run_metadata.json",OUT/"phase5_results.json"
    if a.resume:
        if not mp.exists() or not rp.exists(): raise FileNotFoundError("Phase 5 resume files missing.")
        if json.loads(mp.read_text())!=meta: raise ValueError("Phase 5 settings differ.")
        rows=json.loads(rp.read_text())
    else:
        if mp.exists() or rp.exists(): raise FileExistsError("Phase 5 outputs exist; use --resume.")
        rows=[]; atomic_json(mp,meta)
    completed={r["experiment"] for r in rows if r.get("status")=="completed"}
    for i,c in enumerate(configs,1):
        n=c["name"]
        if n in completed: print("[SKIP]",n); continue
        start=time.time(); print(f"\nPHASE 5 {i}/5: {n}")
        try:
            best_dir=ROOT/"checkpoints/experiments"/n/"best"
            existing_checkpoint=tf.train.latest_checkpoint(str(best_dir))
            if existing_checkpoint:
                print(f"[REUSE] Found trained checkpoint; skipping training: {existing_checkpoint}")
                h=None
            else:
                h=run_experiment(experiment_name=n,model_type="full",encoder_type="gru",
                    use_cross_modal_attention=False,pretrained=c["pretrained"],
                    unfreeze_last_block=c["unfreeze_last_block"],train_all_backbone=False,
                    caption_count=c["caption_count"],image_augmentation=c["image_augmentation"],
                    negative_sampling_strategy="random",epochs=1,batch_size=BATCH,seed=SEED,fold=FOLD,
                    learning_rate=LR,temperature=TEMP,loss_type="info_nce",output_dim=OUTPUT_DIM,
                    dropout=DROPOUT,log_dir="runs/phase5_data_strategy",
                    max_train_steps_per_epoch=TRAIN_STEPS,max_validation_steps=VAL_STEPS)
            m=evaluate_full_checkpoint(n,encoder_type="gru",pretrained=c["pretrained"],
                unfreeze_last_block=c["unfreeze_last_block"],batch_size=EVAL_BATCH_SIZE)
            losses=(h.get("val_retrieval_loss") or h["val_loss"]) if h is not None else []
            rows.append({"experiment":n,"status":"completed","seed":SEED,"batch_size":BATCH,
                "train_steps":TRAIN_STEPS,"validation_steps":VAL_STEPS,
                "caption_count":c["caption_count"],"image_augmentation":c["image_augmentation"],
                "pretrained":c["pretrained"],"unfreeze_last_block":c["unfreeze_last_block"],
                "best_validation_retrieval_loss":min(losses) if losses else None,
                "overall_validation_mean_recall":m["overall_validation_mean_recall"],
                "overall_validation_mrr":m["overall_validation_mrr"],
                "training_time_minutes":(time.time()-start)/60 if h is not None else None,
                "checkpoint_path":m["checkpoint_path"],
                "evaluation_split":"validation","test_set_used":False})
        except Exception as e:
            rows.append({"experiment":n,"status":"failed","error":str(e)[:1500]}); traceback.print_exc()
        finally:
            tf.keras.backend.clear_session(); gc.collect(); save(OUT,rows)
    save(OUT,rows)

if __name__=="__main__": main()
