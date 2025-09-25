# model.py
import os
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    
except Exception as e:
    YOLO = None

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class YoloDetector:
    def __init__(self, modelPath: str = "models/yolov8.pt", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        if YOLO is None:
            raise RuntimeError("ultralytics package is required for YoloDetector")
        if not os.path.exists(modelPath):
            raise FileNotFoundError(f"YOLO checkpoint not found: {modelPath}")
        self.model = YOLO(modelPath)
        self.model.to(self.device)

    def predict(self, frame) -> List[Detection]:
        results = self.model.predict(source=frame, device=self.device, imgsz=640, conf=0.25, verbose=False)
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                cls = int(b.cls.cpu().numpy()) if hasattr(b, "cls") else int(b.cls)
                label = self.model.names[cls]
                conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else float(b.conf)
                x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy()) if hasattr(b, "xyxy") else tuple(map(float, b.xyxy))
                detections.append(Detection(label=label, confidence=conf, bbox=[x1, y1, x2, y2]))
        return detections

    def fineTune(self, dataYamlPath: str, epochs: int = 100, batch: int = 16, lr: float = 1e-3, imgsz: int = 640, saveDir: str = "runs/train"):
        self.model.train(data=dataYamlPath, epochs=epochs, batch=batch, imgsz=imgsz, lr0=lr, project=saveDir)

class EventTokenizer:
    def __init__(self, labelList: List[str], bboxStats: Optional[Dict[str, float]] = None, embedDim: int = 256):
        self.labelList = labelList
        self.labelToIdx = {l: i for i, l in enumerate(labelList)}
        self.embedDim = embedDim
        self.bboxStats = bboxStats or {"w": 1.0, "h": 1.0}
        self.labelEmbedding = nn.Embedding(len(labelList) + 1, embedDim // 2)
        self.bboxLinear = nn.Linear(4, embedDim // 2)
        self.finalLinear = nn.Linear(embedDim, embedDim)

    def to(self, device):
        self.labelEmbedding = self.labelEmbedding.to(device)
        self.bboxLinear = self.bboxLinear.to(device)
        self.finalLinear = self.finalLinear.to(device)
        return self

    def tokenizeDetections(self, detections: List[Detection], frameSize: Optional[List[int]] = None) -> torch.Tensor:
        device = next(self.finalLinear.parameters()).device
        if len(detections) == 0:
            return torch.zeros(1, self.embedDim, device=device)
        labelsIdx = []
        bboxes = []
        for d in detections:
            labelsIdx.append(self.labelToIdx.get(d.label, len(self.labelList)))
            x1, y1, x2, y2 = d.bbox
            if frameSize:
                fw, fh = frameSize
                x1 /= fw
                y1 /= fh
                x2 /= fw
                y2 /= fh
            bboxes.append([x1, y1, x2, y2])
        labelsIdxT = torch.tensor(labelsIdx, dtype=torch.long, device=device)
        bboxesT = torch.tensor(bboxes, dtype=torch.float32, device=device)
        labelEmb = self.labelEmbedding(labelsIdxT)
        bboxEmb = F.relu(self.bboxLinear(bboxesT))
        concat = torch.cat([labelEmb, bboxEmb], dim=-1)
        out = self.finalLinear(concat)
        out = out.mean(dim=0, keepdim=True)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embedDim: int, numHeads: int, ffDim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedDim, numHeads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embedDim)
        self.norm2 = nn.LayerNorm(embedDim)

        self.ff = nn.Sequential(
            nn.Linear(embedDim, ffDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffDim, embedDim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attnMask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attnMask)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

class SequenceTransformer(nn.Module):
    def __init__(self, embedDim: int = 256, numLayers: int = 4, numHeads: int = 8, ffDim: int = 1024, numClasses: int = 2, dropout: float = 0.1):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            EncoderLayer(embedDim, numHeads, ffDim, dropout),
            num_layers=numLayers
        )
        self.classifier = nn.Linear(embedDim, numClasses)
        self.embeddingNorm = nn.LayerNorm(embedDim)

    def forward(self, seqEmbeddings: torch.Tensor) -> torch.Tensor:
        x = self.embeddingNorm(seqEmbeddings)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class HybridModel:
    def __init__(self, yoloPath: str = "models/yolov8.pt", labelList: Optional[List[str]] = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.detector = YoloDetector(modelPath=yoloPath, device=self.device)
        self.labelList = labelList or ["person", "helmet", "gloves", "goggles", "machine"]
        self.tokenizer = EventTokenizer(self.labelList, embedDim=256).to(self.device)
        self.transformer = SequenceTransformer(embedDim=256, numLayers=4, numHeads=8, ffDim=1024, numClasses=3).to(self.device)
        self.seqBuffer: List[torch.Tensor] = []

    def inferFrame(self, frame, frameSize: Optional[List[int]] = None) -> Dict[str, Any]:
        detections = self.detector.predict(frame)
        embedding = self.tokenizer.tokenizeDetections(detections, frameSize=frameSize).to(self.device)
        self.seqBuffer.append(embedding)
        return {"detections": detections, "embedding": embedding}

    def inferSequence(self, maxLen: int = 64) -> torch.Tensor:
        if len(self.seqBuffer) == 0:
            empty = torch.zeros(1, 1, 256, device=self.device)
            return self.transformer(empty)
        
        seq = torch.stack(self.seqBuffer[-maxLen:], dim=0)
        seq = seq.squeeze(1)
        seq = seq.unsqueeze(0)

        seq = seq.to(self.device)

        with torch.no_grad():
            out = self.transformer(seq)
        return out

    def trainTransformer(self, trainDataset: Dataset, valDataset: Optional[Dataset] = None, epochs: int = 50, batchSize: int = 8, lr: float = 1e-4):
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=self._collateSeq)
        
        opt = torch.optim.AdamW(self.transformer.parameters(), lr=lr)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        
        lossFn = nn.CrossEntropyLoss()
        self.transformer.train()

        for epoch in range(epochs):
            for seqBatch, labels in trainLoader:
                seqBatch = seqBatch.to(self.device)
                labels = labels.to(self.device)
                logits = self.transformer(seqBatch)
                loss = lossFn(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sc.step()

    def _collateSeq(self, batch):
        seqs = []
        labels = []
        maxLen = max(len(s[0]) for s in batch)
        for seq, label in batch:
            padded = torch.zeros(maxLen, 256)
            padded[:len(seq), :] = torch.stack(seq, dim=0).squeeze(1)
            seqs.append(padded)
            labels.append(label)
        seqsT = torch.stack(seqs, dim=0)
        labelsT = torch.tensor(labels, dtype=torch.long)
        return seqsT, labelsT

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.transformer.state_dict(), os.path.join(path, "transformer.pt"))
        torch.save(self.tokenizer.finalLinear.state_dict(), os.path.join(path, "tokenizer_final_linear.pt"))

    def load(self, path: str):
        transformerPath = os.path.join(path, "transformer.pt")
        if os.path.exists(transformerPath):
            self.transformer.load_state_dict(torch.load(transformerPath, map_location=self.device))
        tokPath = os.path.join(path, "tokenizer_final_linear.pt")
        if os.path.exists(tokPath):
            self.tokenizer.finalLinear.load_state_dict(torch.load(tokPath, map_location=self.device))