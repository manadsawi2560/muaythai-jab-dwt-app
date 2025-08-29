# MuayThai Jab DWT App — Tempo‑Invariant (v2)

แอปสำหรับวิเคราะห์ท่า **Jab (หมัดซ้าย)** จากวิดีโอ **โค้ช** และ **ผู้เรียน** ด้วย **BlazePose (33 จุด)** + **DTW alignment** เน้นความถูกต้องของ “รูปทรงท่า” โดย **ไม่คิดความเร็ว/ความช้า** ของการออกหมัด (tempo‑invariant) พร้อมระบบให้คะแนนรายส่วน (หมัด/แก่นร่าง/เท้า), หาเฟรม **impact**, ส่งออกวิดีโอ **ซ้าย–ขวา** และมี **Frame‑by‑Frame Viewer** สำหรับดูทีละเฟรมในหน้าเว็บแอป

> เวอร์ชันนี้เป็นผลจากการแก้ไข ปรับปรุง และดีบั๊กหลายรอบ (ดู Changelog ท้ายไฟล์)

---

## 1) คุณสมบัติเด่น
- 🔹 **Tempo‑Invariant**: ไม่พึ่งพาฟีเจอร์ความเร็วทุกชนิด (ไม่มี velocity) และใช้ **Dynamic Time Warping (DTW)** จับคู่ลำดับเฟรม ทำให้เปรียบเทียบกันได้แม้ผู้เรียนจะทำเร็ว/ช้า
- 🔹 **แบ่ง 3 ส่วนร่างกาย**:  
  **Punch** (ข้อมือ–ศอก–ไหล่), **Core** (ไหล่–สะโพก–แนวลำตัว), **Feet** (สะโพก–เข่า–ข้อเท้า)
- 🔹 **Impact Detection**: เลือกเฟรมที่ระยะ **ไหล่–ข้อมือซ้าย** มากที่สุดของโค้ชเป็นจุด Impact แล้วแม็ปไปยังผู้เรียนตาม DTW path
- 🔹 **ให้คะแนนรายส่วน** + คะแนนรวม (0–100) จาก **RMSE หลัง robust z‑score** (ใช้สถิติของโค้ชกับทั้งสองฝั่งเพื่อความยุติธรรม)
- 🔹 **วิดีโอผลลัพธ์จัดวางแบบซ้าย–ขวา** พร้อมแสดง **คะแนนซ้อนบนวิดีโอ**
- 🔹 **Frame‑by‑Frame Viewer** ในหน้าเว็บ: แสดงภาพซ้าย–ขวาพร้อม skeleton, ปรับความกว้างภาพได้, มี **ปุ่ม Prev/Next/First/Last** และเลือก step (±1/±5)
- 🔹 รองรับ Windows/macOS/Linux (รันด้วย Python + Streamlit)

---

## 2) โครงสร้างโปรเจกต์
```plaintext
muaythai-jab-dwt-app/
├─ app.py                      # Streamlit UI (upload, run, viewer, controls)
├─ requirements.txt
├─ README.md                   # ไฟล์นี้
├─ configs/
│  └─ jab_left.yaml           # ค่ากำหนดท่า jab, สี, น้ำหนักคะแนน, tolerance, render
├─ uploads/                   # วิดีโออัปโหลด (เก็บถาวร)
├─ outputs/                   # ผลลัพธ์ประมวลผล
│  ├─ coach_landmarks.csv
│  ├─ student_landmarks.csv
│  ├─ coach_meta.json
│  ├─ student_meta.json
│  ├─ analysis.json           # dtw_cost, path, impact, scores, ฯลฯ
│  └─ side_by_side_analysis.mp4
└─ src/
   ├─ pose_extractor.py       # BlazePose → CSV/JSON
   ├─ features.py             # ฟีเจอร์แบบไม่ใช้ความเร็ว + robust scaling
   ├─ dtw_utils.py            # DTW (numpy)
   ├─ scoring.py              # rmse→score, impact, mapping
   ├─ renderer.py             # วาด skeleton + render วิดีโอซ้าย–ขวา (preload frames)
   └─ pipeline.py             # ขั้นตอน end‑to‑end (เรียกจาก app/CLI)
```

> หมายเหตุ: โฟลเดอร์ `uploads/` จะถูกสร้างอัตโนมัติเมื่อรันแอปครั้งแรก และเก็บวิดีโอต้นฉบับที่ผู้ใช้อัปโหลด ส่วน `outputs/` เก็บผลลัพธ์ของรุ่นล่าสุด (อาจถูกเขียนทับเมื่อวิเคราะห์ใหม่)

---

## 3) การติดตั้ง
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
> แนะนำ Python 3.9–3.11, กล้อง/วิดีโอ 720p+ จะได้ผลลัพธ์ดีขึ้น

**requirements.txt**
```bash
mediapipe>=0.10.14
opencv-python>=4.10.0
numpy>=1.24
pandas>=2.0
PyYAML>=6.0
streamlit>=1.36
scipy>=1.11
tqdm>=4.66
```

---

## 4) การใช้งาน (โหมดเว็บแอป)
รันด้วย Streamlit:
```bash
streamlit run app.py
```
ขั้นตอนบนหน้าเว็บ:
1. อัปโหลดวิดีโอ **โค้ช** และ **ผู้เรียน** (รองรับ mp4/mov/mkv) → ระบบจะเซฟไว้ใน `uploads/`
2. กด **เริ่มวิเคราะห์** → ระบบจะทำงานเต็ม pipeline และสร้างวิดีโอ + ไฟล์ผลลัพธ์ใน `outputs/`
3. ส่วนผลลัพธ์:
   - **วิดีโอซ้าย–ขวา** (มีคะแนนทับบนคลิป) — ถ้าเบราว์เซอร์เล่นไม่ได้ ให้ใช้ **Frame‑by‑Frame Viewer** แทน
   - **Frame‑by‑Frame Viewer**: ดูภาพทีละคู่เฟรมที่จัดแนวโดย DTW, มีปุ่ม **Prev/Next/First/Last** และเลือก step 1/5, ปรับความกว้างภาพได้จาก Sidebar
4. ปุ่ม **ดาวน์โหลดวิดีโอ** จะปรากฏใต้ตัวเล่น หากไฟล์ถูกเรนเดอร์สำเร็จ

---

## 5) ใช้ผ่าน Command Line (ทางเลือก)
```bash
python -m src.pipeline --coach uploads/coach_example.mp4 \
                       --student uploads/student_example.mp4 \
                       --cfg configs/jab_left.yaml \
                       --out outputs
```
จะได้ `outputs/analysis.json` และ `outputs/side_by_side_analysis.mp4`

---

## 6) รายละเอียดอัลกอริทึม
**6.1 Pose & Normalize**
- ใช้ **BlazePose** หา landmark 33 จุด/เฟรม → บันทึกเป็น CSV (`frame,id,x,y,z,visibility`)
- Normalize: origin = **mid‑hip** (จุดกึ่งกลางสะโพกซ้าย/ขวา), scale = **hip width** ต่อเฟรม

**6.2 ฟีเจอร์ (ไม่พึ่งพาความเร็ว)**
- **Punch**: มุมข้อศอก (ไหล่–ศอก–ข้อมือ), ระยะไหล่–ข้อมือซ้าย
- **Core**: torsion (แนวไหล่ vs สะโพก), lean (เอนลำตัวจากแนวตั้ง), ความยาวลำตัว
- **Feet**: ความกว้างฐานเท้า (ระยะข้อเท้าซ้าย–ขวา), ผลรวมองศางอเข่าซ้าย+ขวา

**6.3 Robust z‑score**
- ฟีเจอร์แต่ละช่องทำ robust z‑score:  
  \( z = (x - \text{median}) / (1.4826 * \text{MAD}) \)
- เพื่อความยุติธรรม: **ใช้สถิติของโค้ช** กับทั้งโค้ชและผู้เรียน (scaling เหมือนกันทั้งสองฝั่ง)

**6.4 DTW (Dynamic Time Warping)**
- จัดแนวลำดับด้วย DTW บน **global feature** (concat ของ Punch/Core/Feet) → ได้ path ของคู่เฟรม \((i,j)\)
- ทำให้ไม่สนความเร็ว/ความช้า (tempo‑invariant)

**6.5 Impact**
- เลือกเฟรมโค้ชที่ **ระยะไหล่–ข้อมือซ้าย** มากที่สุดเป็น **Impact** → แม็ปไปยังผู้เรียนด้วย DTW path

**6.6 การให้คะแนน**
- สำหรับแต่ละส่วน (punch/core/feet):
  1) นำฟีเจอร์ของโค้ช/ผู้เรียนที่ถูกจัดแนวด้วย DTW มาเทียบกัน
  2) คำนวณ **RMSE** ต่อส่วน
  3) แปลงเป็นคะแนน 0–100 ด้วย tolerance รายส่วน (ใน `configs/jab_left.yaml`)
- รวมเป็น **final score** ด้วยน้ำหนักถ่วง (ค่าเริ่มต้น punch 0.34, core 0.33, feet 0.33)

**6.7 Render (ซ้าย–ขวา + คะแนนบนวิดีโอ)**
- เพื่อหลีกเลี่ยงปัญหา random seeking ของ OpenCV บนบางระบบ:
  **preload** เฉพาะเฟรมที่จำเป็นตาม DTW path แล้วค่อยเรนเดอร์เรียงตาม path
- วาด skeleton แยกสีรายส่วน + แสดง badge คะแนนบนแต่ละฝั่ง (coach/student)

---

## 7) การตั้งค่า (`configs/jab_left.yaml`)
ตัวอย่างค่าที่ใช้:
```yaml
pose:
  model_complexity: 1
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

segments:
  punch: ["LEFT_WRIST","LEFT_ELBOW","LEFT_SHOULDER"]
  core:  ["LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP"]
  feet:  ["LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"]

colors:
  punch: [60, 220, 60]
  core:  [60, 120, 240]
  feet:  [220, 60, 200]
  rest:  [180, 180, 180]

scoring:
  weights:     { punch: 0.34, core: 0.33, feet: 0.33 }
  tolerances:  { punch: 0.35, core: 0.25, feet: 0.25 }

render:
  out_fps: 30
  side_by_side: true
  font_scale: 0.7
  thickness: 2

dtw:
  window: null  # หรือระบุจำนวนเฟรม เช่น 15
```
**ปรับแต่งที่แนะนำ**
- ถ้าคะแนนเป็น 0 ง่ายเกินไป → เพิ่ม tolerance ของส่วนที่เข้มงวดเกิน เช่น `core: 0.50, feet: 0.50`
- อยากคุมคุณภาพ/ความเร็ว BlazePose → เพิ่ม/ลด `model_complexity`

---

## 8) เอาต์พุตที่ได้
- `outputs/analysis.json` — โครงสร้างตัวอย่าง:
```json
{
  "dtw_cost": 76.62,
  "path_len": 53,
  "impact": { "coach": 20, "student": 26 },
  "scores": {
    "punch": {"rmse": 0.27, "score": 21.0},
    "core":  {"rmse": 0.76, "score": 0.0},
    "feet":  {"rmse": 0.54, "score": 0.0},
    "final_score": 7.14
  },
  "path": [[0,0],[1,1], ...],
  "outputs": { "video": "outputs/side_by_side_analysis.mp4" },
  "errors": {}
}
```
- `outputs/side_by_side_analysis.mp4` — คลิปซ้าย–ขวาพร้อมคะแนนบนวิดีโอ
- Landmarks/Meta: `coach_landmarks.csv`, `student_landmarks.csv`, `coach_meta.json`, `student_meta.json`

---

## 9) แนวทางถ่ายทำวิดีโอเพื่อผลลัพธ์ที่ดี
- กล้องนิ่ง, มุมมองคงที่, ความละเอียด ≥ 720p, แสงพอ, ฉากหลังไม่รก
- ตัวแบบอยู่เต็มตัวในเฟรม (หัวจรดเท้า) และไม่ถูกบัง
- **ท่า Jab ซ้าย**: หันด้านซ้ายเข้ากล้องพอประมาณเพื่อให้ landmark เห็นชัด

---

## 10) Troubleshooting / ข้อผิดพลาดที่เคยพบ & การแก้
- **`NoneType` ใน `cfg['pose']`** → ไฟล์ `jab_left.yaml` ไม่ครบ ให้ใช้ไฟล์ตัวอย่างด้านบน
- **`UnboundLocalError: coach_csv`** → ต้องสร้าง path ตัวแปร output ก่อนเรียก `extract_pose` (แก้แล้วใน `pipeline.py`)
- **`local variable 'cv2' referenced before assignment`** ใน renderer → เกิดจาก import cv2 ซ้ำในฟังก์ชัน (แก้แล้ว)
- **วิดีโอออกมาเป็นสีขาวล้วน** → เกิดจาก random seeking; แก้ด้วยการ **preload** เฟรมที่ต้องใช้
- **Streamlit เด้งกลับให้อัปโหลดใหม่** เวลาเลื่อนสไลเดอร์ → เดิมใช้ `TemporaryDirectory`; แก้เป็นเก็บไฟล์ใน `uploads/` และใช้ `st.session_state`
- **คำเตือน `use_column_width`** → เปลี่ยนไปใช้ `width=` หรือ `use_container_width=True` แล้ว (ใน viewer ใช้ `width=`)

---

## 11) ข้อจำกัด & งานต่อยอด
- ยังไม่รองรับหลายท่า (ตอนนี้โฟกัส **Jab ซ้าย** อย่างเดียว) → สามารถเพิ่มโปรไฟล์ท่าอื่นได้ในอนาคต
- คะแนนเป็นแบบสรุปทั้งคลิป (จาก DTW path ทั้งเส้น) — ถ้าต้องการ “คะแนนรายเฟรม” สามารถคำนวณ RMSE ต่อคู่เฟรมแล้วแสดงใน viewer/กราฟเพิ่มได้
- Mediapipe ใช้ CPU ได้สบาย แต่ถ้าวิดีโอหลายไฟล์/ยาวมาก อาจต้องทำคิวงานหรือ batch mode

---

## 12) Changelog (สรุปสิ่งที่ทำไปบ้าง)
- เปลี่ยนเป็น **tempo‑invariant** (ตัดฟีเจอร์ความเร็วทั้งหมด) + ใช้ **DTW**
- ปรับ **impact** เป็น max ระยะไหล่–ข้อมือซ้าย (โค้ช) แล้วแม็ปด้วย DTW
- Robust z‑score ด้วยสถิติของโค้ช ใช้สเกลเดียวกันทั้งสองฝั่ง
- เรนเดอร์ **ซ้าย–ขวา** (เดิมเป็นบน–ล่าง)
- ซ้อน **คะแนน** (final + รายส่วน) บนวิดีโอ
- ปรับ renderer เป็น **preload frames** (เลิกสุ่ม seek) แก้คลิปขาวล้วน
- ทำ **Frame‑by‑Frame Viewer** (ภาพ ไม่ใช่วิดีโอ) พร้อมปุ่ม **Prev/Next/First/Last** และเลือก step 1/5
- ย้ายคอนโทรลไป **Sidebar** และรองรับปรับความกว้างภาพ (แก้ปัญหาภาพใหญ่เกิน)
- เก็บไฟล์อัปโหลดไว้ที่ `uploads/` (ถาวร) และผลลัพธ์ไว้ที่ `outputs/`
- แก้คำเตือน `use_column_width` → ใช้ `width=` แทน
- เพิ่มการตรวจ/ข้อความ error ให้เข้าใจง่ายใน pipeline

---

## 13) License & Credits
- ใช้ **MediaPipe (BlazePose)** ของ Google Research ตามสัญญาอนุญาตของโครงการนั้น ๆ
- โค้ดส่วนอื่น ๆ เผยแพร่เพื่อการวิจัย/ต้นแบบ ปรับใช้ในงานจริงควรทดสอบและตรวจสอบความปลอดภัย/ความเป็นส่วนตัวของข้อมูลเพิ่มเติม

---
