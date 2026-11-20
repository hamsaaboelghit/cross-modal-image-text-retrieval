"""Shared evaluation and safe result-writing helpers for VisionMatch studies."""
import json
from pathlib import Path
import tensorflow as tf
from src.data.flickr30k_dataset import Flickr30kDataset
from src.data.vocabulary import load_vocabulary
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.models.full_model import CrossModalRetrievalModel

ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = ROOT / "data/processed/flickr30k/vocabularies/fold_1_vocab.json"

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
    temp.replace(path)

def evaluate_full_checkpoint(experiment_name, encoder_type="gru",
        cross_attention=False, pretrained=True, unfreeze_last_block=False,
        train_all_backbone=False, output_dim=512, dropout=0.1,
        batch_size=4, checkpoint_subdir="best"):
    vocab = load_vocabulary(VOCAB_PATH)
    model = CrossModalRetrievalModel(
        vocab_size=len(vocab), encoder_type=encoder_type, embedding_dim=300,
        hidden_dim=512, output_dim=output_dim, num_layers=1, dropout=dropout,
        bidirectional=False, use_attention=True, pretrained=pretrained,
        unfreeze_last_block=unfreeze_last_block,
        train_all_backbone=train_all_backbone,
        use_cross_modal_attention=cross_attention)
    model(tf.zeros((1,224,224,3), tf.float32),
          tf.fill((1,5,34), tf.constant(4, tf.int32)), training=False)
    checkpoint_dir = ROOT / "checkpoints/experiments" / experiment_name / checkpoint_subdir
    checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    status = tf.train.Checkpoint(model=model).restore(checkpoint)
    status.assert_existing_objects_matched()
    status.expect_partial()
    val = Flickr30kDataset(fold_number=1, split="validation")
    dataset = val.as_dataset(batch_size=batch_size, shuffle=False,
                             drop_remainder=False, seed=42)
    result = RetrievalEvaluator(model=model, query_batch_size=32,
        candidate_chunk_size=8192, ks=(1,5,10)).evaluate(dataset)
    t2i, i2t = result["text_to_image"], result["image_to_text"]
    return {
        "checkpoint_path": checkpoint, "validation_images": result["num_images"],
        "text_to_image": t2i, "image_to_text": i2t,
        "overall_validation_mean_recall": (t2i["mean_recall"]+i2t["mean_recall"])/2,
        "overall_validation_mrr": (t2i["mrr"]+i2t["mrr"])/2,
        "evaluation_split": "validation", "test_set_used": False}

