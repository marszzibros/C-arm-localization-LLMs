from Inference_localization import Inference
import sys

model_id = sys.argv[1]
mode = sys.argv[2]  # "test" or "classification"
inference = Inference(model_id, mode=mode, file_path=sys.argv[3])
inference.inference()