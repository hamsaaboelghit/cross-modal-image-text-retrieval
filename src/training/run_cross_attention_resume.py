"""Resume Cross-Attention by 500 batches from the original 1,000-batch checkpoint."""
import argparse,os,gc
import tensorflow as tf
from src.training.experiment_runner import run_experiment
from src.training.experiment_utils import evaluate_full_checkpoint

SOURCE="deadline_fast_20261119_cross_attention_fold1_seed42"
OUTPUT="deadline_resume500_20261120_cross_attention_fold1_seed42"

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--list",action="store_true"); a=p.parse_args()
    source_dir=f"checkpoints/experiments/{SOURCE}/latest"
    checkpoint=tf.train.latest_checkpoint(source_dir)
    if checkpoint is None: raise FileNotFoundError(f"No source checkpoint in {source_dir}")
    if a.list:
        print("Source:",checkpoint,"| additional train batches: 500 | validation batches: 200"); return
    out=f"checkpoints/experiments/{OUTPUT}"
    if os.path.exists(out): raise FileExistsError(f"{out} exists. Inspect it; this script never overwrites.")
    try:
        run_experiment(experiment_name=OUTPUT,model_type="full",encoder_type="lstm",
            use_cross_modal_attention=True,pretrained=True,unfreeze_last_block=False,
            train_all_backbone=False,caption_count=5,image_augmentation="none",
            negative_sampling_strategy="random",epochs=2,batch_size=4,seed=42,fold=1,
            learning_rate=1e-4,temperature=0.07,loss_type="info_nce",output_dim=512,dropout=0.1,
            log_dir="runs/deadline_resume500",max_train_steps_per_epoch=500,
            max_validation_steps=200,resume_from=checkpoint,resume_skip_batches=1000)
        metrics=evaluate_full_checkpoint(OUTPUT,encoder_type="lstm",cross_attention=True,
            batch_size=4,checkpoint_subdir="latest")
        print("Resumed Cross-Attention validation Mean Recall:",metrics["overall_validation_mean_recall"])
    finally:
        tf.keras.backend.clear_session(); gc.collect()

if __name__=="__main__": main()
