from Inference_navigation import Inference
import sys

model_id = sys.argv[1]
mode = sys.argv[2]  # "test" or "classification"
inference = Inference(model_id, mode=mode, next_path=sys.argv[3])
inference.inference()