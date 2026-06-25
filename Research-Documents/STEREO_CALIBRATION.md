# Stereo Camera Calibration (FoGaze 2-camera depth)

ทำไมต้อง calibrate: ภาพ depth ที่ออกมาเป็นจุดสีฟ้ารั่วๆ เต็มจอ เกิดจากกล้องสองตัว
ยัง **ไม่ rectify** — จุดเดียวกันในภาพซ้าย/ขวาไม่อยู่แถวเดียวกัน ทำให้ StereoSGBM
จับคู่ไม่ได้ การ calibrate จะแก้เรื่องนี้ และทำให้ระยะที่วัดได้เป็น **เมตริกจริง** (ซม.)
แทนการเดาค่า `focal_px`/`baseline` ด้วยมือ

---

## 1. เตรียมกระดานหมากรุก (chessboard)

- ใช้กระดาน **9 × 6 มุมภายใน** (internal corners) — คือกระดาน 10×7 ช่อง
- ขนาดช่อง **25 มม.** ต่อช่อง (ค่า default `SQUARE_MM = 25.0`)
- ปริ้นต์ใส่กระดาษ A4 แล้ว **ติดบนแผ่นแข็งให้เรียบ** (ห้ามโค้ง/ย่น)
- ถ้าปริ้นต์ขนาดอื่น ให้วัดช่องจริงแล้วแก้ `SQUARE_MM` ใน
  `modules/stereo_calibrator.py` (กระทบสเกลระยะ)

> ขนาดกระดาน/ช่องต้องตรงกับมุมจริง ไม่งั้น `findChessboardCorners` จะหาไม่เจอ

---

## 2. ติดตั้งกล้องให้ถูก

- วางสองกล้อง **ระดับเดียวกัน สูงเท่ากัน หันขนานกัน**
- ห่างกัน (baseline) ~**6 ซม.** (ยิ่งห่างยิ่งวัดไกลได้ดี แต่ของใกล้จะ match ยากขึ้น)
- ยึดให้แน่น **ห้ามขยับหลัง calibrate** — ถ้าขยับต้อง calibrate ใหม่
- ใช้ความละเอียดเดียวกับตอนรันจริง (default 640×480)

---

## 3. รัน calibration

```bash
cd /media/q/SaveFile/Projects/SchoolProject/KAITO-Kanagawa_exchange/FoGaze
python3 modules/stereo_calibrator.py -l 0 -r 2
```

(`-l` = index กล้องซ้าย, `-r` = index กล้องขวา — ดู index ได้จากการรัน
`python3 modules/stereo_depth_estimator.py` แล้วมันจะ scan ให้)

หน้าต่างพรีวิวจะแสดงภาพซ้าย|ขวาคู่กัน:

| ปุ่ม | ทำอะไร |
|------|--------|
| `SPACE` | จับภาพคู่ปัจจุบัน (ได้เฉพาะตอนเจอกระดานใน **ทั้งสองภาพ** → ป้ายขึ้น `board:BOTH` สีเขียว) |
| `c` | คำนวณ calibration จากภาพที่จับไว้ แล้วเซฟ |
| `u` | ลบภาพคู่ล่าสุด (undo) |
| `q` | ออก |

### วิธีจับภาพให้ดี (สำคัญที่สุด)

ถือกระดานให้กล้อง **เห็นครบทั้งสองตัว** แล้วกด SPACE เก็บหลายๆ มุม:

- เก็บ **อย่างน้อย 15–20 คู่** (ขั้นต่ำของโค้ดคือ 8 คู่)
- ขยับกระดานให้ครอบคลุม **ทุกมุมจอ** — ซ้าย ขวา บน ล่าง กลาง
- **เอียง** กระดานหลายองศา (ก้ม/เงย/ตะแคง) ไม่ใช่ตั้งฉากอย่างเดียว
- เปลี่ยน **ระยะ** ใกล้-ไกล
- กระดานต้องนิ่ง ไม่เบลอ แสงสว่างพอ

จากนั้นกด `c` โปรแกรมจะพิมพ์:

```
[StereoCalibrator] Stereo RMS reprojection error: 0.42 px (<1.0 is good, >2.0 means recapture)
[StereoCalibrator] baseline=6.05cm focal=512.3px
[StereoCalibrator] Saved -> ~/.cache/fogaze3/calib/stereo_0_2.json
```

- **RMS < 1.0 px** = ดี ใช้ได้
- **RMS > 2.0 px** = ไม่ดี กด `u` ลบคู่แย่ๆ หรือเก็บใหม่ให้หลากหลายขึ้น
- เช็ค `baseline` ที่ได้ ควรใกล้ระยะจริงที่วัดด้วยไม้บรรทัด

---

## 4. ตรวจผล

รัน depth view อีกครั้ง — มันจะโหลด calibration อัตโนมัติ:

```bash
python3 modules/stereo_depth_estimator.py -l 0 -r 2
# [StereoDepthEstimator] Using stereo calibration (rectified + metric depth).
```

depth map ควรเป็นโซนสีต่อเนื่องตามวัตถุ (ไม่ใช่จุดรั่วเต็มจอ) และตัวเลข
`center: __ cm` ตรงกลางควรใกล้ระยะจริง ลองเอาวัตถุมาวางหน้ากล้องแล้ววัดเทียบ

ถ้ายังมั่ว → กลับไปเก็บภาพ calibration ใหม่ให้หลากหลายมุมขึ้น หรือเช็คว่ากล้อง
ขยับหลัง calibrate หรือเปล่า

---

## ไฟล์ที่เกี่ยวข้อง

| ไฟล์ | หน้าที่ |
|------|---------|
| `modules/stereo_calibrator.py` | เครื่องมือ calibrate + คลาส `StereoCalibrator` (โหลด/rectify) |
| `modules/stereo_depth_estimator.py` | โหลด calibration อัตโนมัติ → rectify ก่อนทำ SGBM |
| `~/.cache/fogaze3/calib/stereo_<L>_<R>.json` | ผลลัพธ์ calibration (ต่อคู่กล้อง) |

> ไฟล์ผูกกับ **คู่ index + ความละเอียด** ที่ calibrate ถ้าเปลี่ยนกล้อง/พอร์ต/
> ความละเอียด ต้อง calibrate ใหม่ ลบไฟล์ JSON เพื่อเริ่มใหม่ก็ได้
