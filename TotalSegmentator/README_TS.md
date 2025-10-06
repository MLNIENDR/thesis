# CT-Rekonstruktions- und Kalibrierpipeline

Diese Pipeline rekonstruiert ein CT-Volumen aus Projektionsdaten, kalibriert es in Hounsfield-Einheiten (HU), und führt anschließend eine automatische anatomische Segmentierung durch.

---

## Ablauf

### 1. FDK-Rekonstruktion (`recon_fdk_rtk.py`)
Rekonstruiert das µ-Volumen (lineare Schwächung) mit ITK-RTK.  
Speichert:
- `ct_recon_rtk.nii` – internes Format (X,Z,Y)
- `ct_recon_rtk_RAS.nii` – **kanonisches Format (X,Y,Z)** → für alle weiteren Schritte

### 2. µ → HU-Kalibrierung (`calibrate_mu_to_hu.py`)
Wandelt µ in HU um.  
Optionen:
- `--roi-air` und `--roi-water`: Masken aus Phantom
- `--target-tissue-hu`: 0 (Wasser) oder 40 (Weichteil)
- Automodus, falls keine ROIs vorhanden sind (schätzt selbst Luft/Gewebe)

Ausgabe:
- `*_HU_calib.nii.gz` – HU-Volumen
- `ts_preview/coronal_AP_HU_calib.png` – korrekte coronale Ansicht (800×500 mm)
- Optional JSON-Report mit Kalibrierparametern

### 3. Segmentierung (`hu_seg_and_preview.py`)
Prüft HU-Werte auf Plausibilität (min/p50/p99).  
Startet TotalSegmentator (`--ml`) und legt Multi-Label-Segmentierungen an.  
Erzeugt Overlay:
- `ts_preview/coronal_AP_overlay.png`

---

## Typische HU-Werte

| Gewebe | Erwarteter HU |
|:--|--:|
| Luft | −1000 |
| Fett | −100 … −50 |
| Wasser | 0 |
| Muskel | 40 |
| Knochen | 300 – 2500 |

---

## Hinweise & Stolpersteine
- **u/v-Verwechslung:** Falsches Seitenverhältnis durch Detektorachsen-Tausch. Jetzt fest definiert (width ↔ nc, height ↔ nr).
- **Achsenchaos:** RAS-Datei sorgt für konsistente Orientierung über alle Skripte.
- **µ→HU-Kalibrierung:** Ohne Phantom nur Näherung – daher Kalibrierung mit ROI-Phantom dringend empfohlen.
- **HU-Range-Check:** Immer `min≈−1000`, `p50≈0±100`, `p99≈2500±500`.

---

## Schnellstart

```bash
./run_all.sh \
  --mat ct_proj_stack.mat \
  --proj proj.txt \
  --prefix ct_recon_rtk \
  --roi-air phantom_air_mask.nii.gz \
  --roi-water phantom_water_mask.nii.gz \
  --target 0 \
  --seg-dir ts_output