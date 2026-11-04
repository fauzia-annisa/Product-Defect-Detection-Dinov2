import io, base64, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter
from fastapi import FastAPI, UploadFile, File
import matplotlib; matplotlib.use("Agg")
from matplotlib import cm

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("patchcore_dino.pt")
bank, thr, SIZE = ck["bank"].to(device), ck["thr"], ck["size"]
G = SIZE // 14
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
tf = T.Compose([T.Resize((SIZE,SIZE)), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
app = FastAPI(title="Product Defect Detector (DINOv2 + PatchCore)")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    f = model.forward_features(tf(img).unsqueeze(0).to(device))["x_norm_patchtokens"][0]
    d = torch.cdist(f, bank).min(dim=1).values.reshape(1,1,G,G)
    amap = F.interpolate(d, size=img.size[::-1], mode="bilinear")[0,0].cpu().numpy()
    amap = gaussian_filter(amap, 4)
    s = float(amap.max())
    norm = np.clip((amap - thr * 0.6) / (thr * 0.9), 0, 1)
    heat = (cm.jet(norm)[..., :3] * 255).astype(np.uint8)
    over = (0.5*np.array(img) + 0.5*heat).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(over).save(buf, "PNG")
    return {"anomaly_score": s, "threshold": thr, "is_defect": s > thr,
            "heatmap_png_base64": base64.b64encode(buf.getvalue()).decode()}

@app.get("/health")
def health(): return {"status": "ok"}
