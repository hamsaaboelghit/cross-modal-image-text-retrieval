
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 55)
print("PACKAGE IMPORT TEST")
print("=" * 55)

# 1. Package imports
import src
import src.data
import src.models
import src.training
import src.evaluation
import src.retrieval

print("Package imports: PASS")

# 2. Data module
from src.data.flickr30k_dataset import Flickr30kDataset

print("Dataset import: PASS")

# 3. Model modules
from src.models.attention import AdditiveAttention
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.projection_head import CrossModalProjection
from src.models.full_model import CrossModalRetrievalModel

print("Model imports: PASS")

# 4. Loss modules
from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.triplet_loss import TripletContrastiveLoss
from src.training.margin_loss import MarginBasedContrastiveLoss

print("Loss imports: PASS")

print("=" * 55)
print("ALL PACKAGE IMPORT TESTS PASSED!")
print("=" * 55)
