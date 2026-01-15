import os


os.environ["UNSLOTH_DISABLE_PATCHING"] = "1"  
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # disables compile wrappers, keeps other speedups
os.environ["TORCH_COMPILE_DISABLE"] = "1"    # or use TORCH_COMPILE_DEBUG=1 if you want tracing info
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


from unsloth import FastVisionModel 
import pandas as pd
from transformers import TextIteratorStreamer
import threading
from PIL import Image
import os
import time



class Inference:
    def __init__(self, model_id, mode="test", next_path=""):   
        self.model_id = model_id
        self.landmarks = {
            1: "Skull",
            2: "Right humeral head",
            3: "Left humeral head",
            4: "Right scapular", 
            5: "Left scapular",
            6: "Right elbow",
            7: "Left elbow",
            8: "Right wrist",
            9: "Left wrist",
            10: "T1", 
            11: "Carina",
            12: "Right hemidiaphragm",
            13: "Left hemidiaphragm", 
            14: "T12"
        }
        os.system(f"mkdir navigation")
        self.mode = mode
        if mode == "test":
            self.folder = "navigation/first_pred"
            os.system(f"mkdir {self.folder}")
            self.df = pd.read_csv("../../data/navigation_dataset.csv")
            # per patient, sample 256 images
            with open("../prompts/navigation.txt", "r") as f:
                self.system_prompt = f.read()
            self.vertical_flip = False
        elif "flip" in mode:

            self.folder = "navigation_flipped"
            os.system(f"mkdir {self.folder}")
            self.folder = "navigation_flipped/third_pred"
            os.system(f"mkdir {self.folder}")
            self.df = pd.read_csv(next_path)
            # per patient, sample 256 images
            with open("../prompts/navigation.txt", "r") as f:
                self.system_prompt = f.read()
            self.vertical_flip = True
        else:
            self.folder = mode
            os.system(f"mkdir {self.folder}")
            self.df = pd.read_csv(next_path)
            # per patient, sample 256 images
            with open("../prompts/navigation.txt", "r") as f:
                self.system_prompt = f.read()
            self.vertical_flip = False            

        self.model, self.processor = FastVisionModel.from_pretrained(
            model_name=self.model_id,
            load_in_4bit=False,
            device_map="balanced", 
            max_seq_length=8192,
        )
        FastVisionModel.for_inference(self.model)
        print("Model and processor loaded.")




    def inference(self):
        csv_path = f"{self.model_id}.csv"
        csv_path = csv_path.replace("/", "_")
        csv_path = os.path.join(self.folder, csv_path)
        file_exists = os.path.exists(csv_path)

        for i, row in self.df.iterrows():
            try:
                image_path = row['file_path']

                image = Image.open(image_path).convert("RGB").resize((512, 512))
                if self.vertical_flip:
                    image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
                
                if "first" in self.mode:
                    user_prompt = (
                        "Can you navigate to the **<target>:<target_name>** given the provided X-ray image?"
                        .replace("<target>", str(row['target']))
                        .replace("<target_name>", self.landmarks[row['target']])
                    )
                else:
                    with open("prompts/navigation_multi_step.txt", "r") as f:
                        user_prompt = f.read()
                    user_prompt = user_prompt.replace("<LANDMARK>", row['landmark'])
                    user_prompt = user_prompt.replace("<X_DIR>", row['x_dir'])
                    user_prompt = user_prompt.replace("<Y_DIR>", row['y_dir'])
                    user_prompt = user_prompt.replace("<X_EXT>", row['x_ext'])
                    user_prompt = user_prompt.replace("<Y_EXT>", row['y_ext'])
                    user_prompt = user_prompt.replace("<TARGET>", str(row['target']))
                    user_prompt = user_prompt.replace("<TARGET_NAME>", self.landmarks[row['target']])

                message = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},  # or image_url=...
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]



                input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
                inputs = self.processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("cuda")


                # --- measure start time ---
                start_time = time.time()
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=1, top_p=0.95, top_k=64,
                )

                generated_text = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

                # --- record end time & compute duration ---
                end_time = time.time()
                elapsed = round(end_time - start_time, 2)  # seconds (2 decimal places)

                # --- create one-row DataFrame ---
                df_row = pd.DataFrame([{
                    "filename": image_path,
                    "output": generated_text,
                    "inference_time_sec": elapsed,
                    "target"    : row['target']
                    
                }])

                # --- append to CSV ---
                df_row.to_csv(
                    csv_path,
                    mode='a',
                    header=not file_exists,
                    index=False,
                    encoding="utf-8"
                )
                file_exists = True

                print(f"✅ Saved output for {image_path} ({elapsed}s)")
            except:
                continue