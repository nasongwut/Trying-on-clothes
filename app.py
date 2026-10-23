"""
ห้องลองชุด AI — เว็บแอป Python (Flask) + FASHN VTON v1.5 (รันในเครื่อง)

โฟลเดอร์:
  wardrobe/<male|female>/<top|shorts|pants|skirt|dress>/   รูปเสื้อผ้า (วางไฟล์เองก็ได้)
  results/                                                ภาพที่ AI สร้าง
เปิดใช้งาน: python app.py  แล้วเข้า http://127.0.0.1:8765
"""
import io
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # op ที่ Mac GPU ไม่รองรับ → ใช้ CPU แทน
HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / "hf-cache"))

from flask import Flask, abort, jsonify, render_template, request, send_from_directory  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

WARDROBE = HERE / "wardrobe"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
PORT = int(os.environ.get("TRYON_PORT", "8765"))
MOCK = os.environ.get("FASHN_MOCK") == "1"  # ทดสอบหน้าเว็บโดยไม่โหลดโมเดล

CATS = {
    "male": {"label": "ชาย", "items": [["top", "เสื้อ"], ["shorts", "กางเกงขาสั้น"], ["pants", "กางเกงขายาว"]]},
    "female": {"label": "หญิง", "items": [["top", "เสื้อ"], ["shorts", "กางเกงขาสั้น"], ["pants", "กางเกงขายาว"],
                                          ["skirt", "กระโปรง"], ["dress", "เดรส"]]},
}
# หมวดในเว็บ → หมวดของ FASHN VTON
FASHN_CAT = {"top": "tops", "shorts": "bottoms", "pants": "bottoms", "skirt": "bottoms", "dress": "one-pieces"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

for g, info in CATS.items():
    for c, _ in info["items"]:
        (WARDROBE / g / c).mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# ---------------- AI model ----------------
state = {"status": "loading", "device": None, "error": None}
pipe = None
pipe_lock = threading.Lock()


def pick_device():
    if os.environ.get("TRYON_DEVICE"):
        return os.environ["TRYON_DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_for_mps():
    """Mac GPU (MPS) ไม่รองรับ float64 ซึ่งโมเดลใช้ตอนคำนวณ RoPE
    → คำนวณส่วนนี้บน CPU (ค่าเหมือนเดิมทุกประการ) แล้วย้ายผลกลับไป GPU"""
    from fashn_vton import tryon_mmdit
    if getattr(tryon_mmdit.rope, "_mps_patched", False):
        return
    orig = tryon_mmdit.rope

    def rope(pos, dim, theta):
        if pos.device.type == "mps":
            return orig(pos.cpu(), dim, theta).to(pos.device)
        return orig(pos, dim, theta)

    rope._mps_patched = True
    tryon_mmdit.rope = rope


def load_model():
    global pipe
    try:
        if MOCK:
            state.update(status="ready", device="mock")
            return
        patch_for_mps()
        from fashn_vton import TryOnPipeline
        dev = pick_device()
        print(f"[AI] กำลังโหลดโมเดลบน {dev} ...", flush=True)
        pipe = TryOnPipeline(weights_dir=str(WEIGHTS), device=dev)
        state.update(status="ready", device=dev)
        print(f"[AI] พร้อมใช้งาน ({dev})", flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        state.update(status="error", error=str(e))


def run_one(person, garment, category, photo_type, steps, seed):
    global pipe
    if MOCK:  # ภาพจำลองสำหรับทดสอบ
        time.sleep(0.2)
        out = person.copy()
        g = garment.copy()
        g.thumbnail((person.width // 2, person.height // 3))
        out.paste(g, (person.width // 4, person.height // (2 if category == "bottoms" else 4)))
        return out
    kw = dict(person_image=person, garment_image=garment, category=category,
              garment_photo_type=photo_type, num_timesteps=steps, seed=seed)
    try:
        return pipe(**kw).images[0]
    except Exception:
        if state["device"] != "mps":
            raise
        print("[AI] Mac GPU ทำงานไม่ได้กับโมเดลนี้ → สลับไปใช้ CPU (ช้ากว่า)", flush=True)
        traceback.print_exc()
        from fashn_vton import TryOnPipeline
        pipe = TryOnPipeline(weights_dir=str(WEIGHTS), device="cpu")
        state["device"] = "cpu"
        return pipe(**kw).images[0]


# ---------------- helpers ----------------
def open_img(fileobj, max_side=1536):
    img = Image.open(fileobj)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def valid(gender, cat):
    return gender in CATS and cat in [c for c, _ in CATS[gender]["items"]]


def list_clothes():
    out = []
    for g, info in CATS.items():
        for c, _ in info["items"]:
            for p in (WARDROBE / g / c).iterdir():
                if p.suffix.lower() in IMG_EXT and not p.name.startswith("."):
                    out.append({"id": f"{g}/{c}/{p.name}", "gender": g, "cat": c,
                                "url": f"/wardrobe/{g}/{c}/{p.name}", "t": p.stat().st_mtime})
    out.sort(key=lambda x: -x["t"])
    return out


# ---------------- routes ----------------
@app.get("/")
def index():
    return render_template("index.html", cats_json=json.dumps(CATS, ensure_ascii=False))


@app.get("/api/health")
def health():
    return jsonify(ok=state["status"] == "ready", model="FASHN VTON v1.5", **state)


@app.get("/api/clothes")
def api_clothes():
    return jsonify(list_clothes())


@app.post("/api/clothes")
def api_add_clothes():
    gender, cat = request.form.get("gender"), request.form.get("cat")
    if not valid(gender, cat):
        abort(400, "หมวดไม่ถูกต้อง")
    saved = 0
    for f in request.files.getlist("files"):
        try:
            img = open_img(f.stream, 1024)
            img.save(WARDROBE / gender / cat / f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}.jpg", quality=92)
            saved += 1
        except Exception as e:  # noqa: BLE001
            print("อ่านรูปไม่ได้:", f.filename, e)
    return jsonify(saved=saved)


@app.delete("/api/clothes/<gender>/<cat>/<name>")
def api_del_clothes(gender, cat, name):
    if not valid(gender, cat) or "/" in name or name.startswith("."):
        abort(400)
    p = WARDROBE / gender / cat / name
    if p.exists():
        p.unlink()
    return jsonify(ok=True)


@app.get("/wardrobe/<gender>/<cat>/<name>")
def wardrobe_file(gender, cat, name):
    if not valid(gender, cat):
        abort(404)
    return send_from_directory(WARDROBE / gender / cat, name)


@app.get("/results/<name>")
def result_file(name):
    return send_from_directory(RESULTS, name)


@app.get("/api/results")
def api_results():
    files = sorted(RESULTS.glob("*.png"), key=lambda p: -p.stat().st_mtime)[:12]
    return jsonify([f"/results/{p.name}" for p in files])


@app.post("/api/tryon")
def api_tryon():
    if state["status"] != "ready":
        msg = "โมเดลกำลังโหลด รอสักครู่แล้วลองใหม่" if state["status"] == "loading" \
            else f"โหลดโมเดลไม่สำเร็จ: {state['error']}"
        return jsonify(error=msg), 503
    if "person" not in request.files:
        return jsonify(error="ยังไม่ได้อัพรูปคน"), 400
    gender = request.form.get("gender")
    ids = json.loads(request.form.get("items", "[]"))
    photo_type = request.form.get("photo_type", "flat-lay")
    steps = max(10, min(50, int(request.form.get("steps", 30))))
    seed = int(request.form.get("seed", 42))

    # ตรวจรายการ + เรียงลำดับ: ท่อนล่างก่อน แล้วค่อยเสื้อ (เสื้อทับกางเกงดูเป็นธรรมชาติกว่า)
    items = []
    for item_id in ids:
        g, c, name = item_id.split("/", 2)
        p = WARDROBE / g / c / name
        if g != gender or not valid(g, c) or not p.exists():
            return jsonify(error=f"ไม่พบเสื้อผ้า: {item_id}"), 400
        items.append((c, p))
    if not items:
        return jsonify(error="ยังไม่ได้เลือกเสื้อผ้า"), 400
    order = {"dress": 0, "shorts": 1, "pants": 1, "skirt": 1, "top": 2}
    items.sort(key=lambda x: order[x[0]])

    try:
        person = open_img(request.files["person"].stream)
        t0 = time.time()
        with pipe_lock:
            for i, (c, p) in enumerate(items):
                print(f"[AI] ชิ้นที่ {i + 1}/{len(items)}: {c} → {FASHN_CAT[c]} ({steps} steps)", flush=True)
                person = run_one(person, open_img(p), FASHN_CAT[c], photo_type, steps, seed)
        name = f"tryon-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}.png"
        person.save(RESULTS / name)
        sec = round(time.time() - t0, 1)
        print(f"[AI] เสร็จใน {sec} วินาที → results/{name}", flush=True)
        return jsonify(url=f"/results/{name}", seconds=sec, device=state["device"])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    threading.Thread(target=load_model, daemon=True).start()
    print(f"[เว็บ] เปิดที่ http://127.0.0.1:{PORT}  (ปิดหน้าต่างนี้เพื่อหยุด)", flush=True)
    app.run(host="127.0.0.1", port=PORT, threaded=True)
