import argparse
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from model import build_model
from dataset import get_transforms
from training_script import load_config

def predict(args):
    # 1. Setup Device and Config
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    
    # 2. Load Model
    model = build_model(cfg).to(dev)
    checkpoint = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 3. Setup Data
    img_size = cfg["data"].get("img_size", 518)
    transform = get_transforms(img_size=img_size, is_train=False)
    label_cols = cfg["data"].get("label_cols", ["Creativity", "Belonging", "EaseOfUse"])
    
    # 4. Find Images
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        return

    if input_path.is_file():
        img_files = [input_path]
    else:
        img_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
    
    if not img_files:
        print(f"No images found in {input_path}")
        return

    # 5. Inference
    results = []
    with torch.no_grad():
        for img_p in tqdm(img_files, desc="Rating Images"):
            try:
                raw_img = Image.open(img_p).convert("RGB")
                img_t = transform(raw_img).unsqueeze(0).to(dev)
                
                preds = model(img_t).cpu().numpy()[0]
                
                res = {"Image": img_p.name}
                for i, col in enumerate(label_cols):
                    res[col] = round(float(preds[i]), 3)
                results.append(res)
            except Exception as e:
                print(f"Error processing {img_p.name}: {e}")

    # 6. Save Results
    if results:
        df = pd.DataFrame(results)
        out_path = Path(args.output)
        df.to_csv(out_path, index=False)
        print(f"\n[done] Predictions saved to: {out_path}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to image or folder")
    parser.add_argument("--output", default="predictions.csv", help="CSV to save results")
    parser.add_argument("--ckpt", default="checkpoints/super_head_run_2026-04-26_07-29-26/best.pt")
    parser.add_argument("--config", default="configs/super_head_training.yaml")
    predict(parser.parse_args())
