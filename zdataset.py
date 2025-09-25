import os
import json
import random
import shutil
from zproperties import DATASET_JSON, IMAGES_DIR, OUTPUT_DIR, SPLIT_RATIO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def coco_to_yolo_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]

def prepare_dataset():
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Mapear categorias para índices contínuos (0..N-1)
    category_map = {cat_id: idx for idx, cat_id in enumerate(categories.keys())}

    # Criar splits train/val
    image_ids = list(images.keys())
    train_ids, val_ids = train_test_split(image_ids, test_size=SPLIT_RATIO, random_state=42)

    splits = {
        "train": train_ids,
        "val": val_ids
    }

    # Criar diretórios de saída
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # Agrupar anotações por imagem
    ann_per_image = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in ann_per_image:
            ann_per_image[img_id] = []
        ann_per_image[img_id].append(ann)

    # Converter e salvar
    for split, ids in splits.items():
        for img_id in tqdm(ids, desc=f"Processando {split}"):
            img_info = images[img_id]
            file_name = img_info["file_name"]
            width, height = img_info["width"], img_info["height"]

            # Copiar imagem
            src_path = os.path.join(IMAGES_DIR, file_name)
            dst_path = os.path.join(OUTPUT_DIR, "images", split, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

            # Criar arquivo de labels
            label_path = os.path.join(OUTPUT_DIR, "labels", split, file_name.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                if img_id in ann_per_image:
                    for ann in ann_per_image[img_id]:
                        cls_id = category_map[ann["category_id"]]
                        yolo_box = coco_to_yolo_bbox(ann["bbox"], width, height)
                        f.write(f"{cls_id} " + " ".join([f"{x:.6f}" for x in yolo_box]) + "\n")

    # Criar data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(OUTPUT_DIR, 'images/train')}\n")
        f.write(f"val: {os.path.join(OUTPUT_DIR, 'images/val')}\n\n")
        f.write(f"nc: {len(categories)}\n")
        f.write(f"names: {list(categories.values())}\n")

    print(f"\n✅ Dataset convertido com sucesso para YOLO em: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_dataset()
