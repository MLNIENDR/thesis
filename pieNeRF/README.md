pieNeRF – Skript-Nachschlagewerk
================================

Kurzer Überblick, was welches Skript macht und wo zentrale Funktionen stecken.


Top-Level
---------
- `train_emission.py`: Mini-Trainingsskript. Kernfunktionen: `parse_args()` (CLI), `poisson_nll()` (Poisson-NLL pro Strahl), `build_pose_rays()` (AP/PA-Rays einmalig auf GPU), `render_minibatch()` (Teilrays rendern, reicht z-Latent + optional CT-Kontext durch), `maybe_render_preview()`/`save_depth_profile()` (volles AP/PA-Render + Profilplots), `sample_act_points()`/`sample_ct_pairs()` (stochastische ACT- bzw. CT-Samples für Zusatz-Loss), `train()` (lädt YAML, baut Dataset/Generator, cached AP/PA-Rays, zieht Zufalls-Indizes, rechnet Poisson-NLL + optionale ACT/CT/z-Reg-Losses, Logging/CSV, Checkpoints, Previews).
- `run_train_emission.sh`: Slurm-Wrapper (Conda aktivieren, `train_emission.py` mit Standardparametern starten, Logs in `slurm/`).
- `configs/spect.yaml`: Basis-Config (Pfad zu `data/manifest.csv`, Bildgeometrie, NeRF-/Sampler-Parameter, Trainings-Chunkgrößen, Attenuation-Flag).
- `test_dataset.py`: Lädt YAML → `get_data()` → druckt Shapes/Min/Max, speichert Quick-Plot (`data_check.png`). Einstieg: `main(config_path="configs/spect.yaml")`.


NeRF-Kern (`nerf/`)
------------------
- `run_nerf_mod.py`: Modifiziertes NeRF für Emission.
  - `run_network(...)`: Flacht Samples, positional encoding, hängt latente Features (z) und ggf. Viewdirs an, achtet auf gemeinsames Device, processed in Chunks.
  - `sample_ct_volume(...)`: Trilineare Interpolation im CT-Würfel (Grid in [-radius, radius]^3 via `grid_sample`).
  - `raw2outputs_emission(...)`: Softplus auf Netz-Output → Emission e(x); berechnet Segmentlängen `dists`, optional Rauschen; bei `use_attenuation` holt μ aus CT, akkumuliert Dämpfung `exp(-∑ μ·Δs)`, gewichtet Emission damit; integriert Line-Integral ∑ e·Δs zu `proj_map`, dazu einfache `disp_map` und `acc_map`; liefert auf Wunsch Debug-Payload (λ/μ/T).
  - `render(...)`: Baut Rays (ggf. aus Pose), optional NDC, hängt near/far + Viewdirs an, broadcastet z-Features pro Ray, ruft `render_rays` chunked auf, reshaped und extrahiert `proj_map`/`disp`/`acc` (oder `rgb_map` im alten Pfad).
  - `render_rays(...)`: Legt z-Samples linear oder in Disparity zwischen near/far, optional jitter (`perturb`), berechnet Weltpunkte, ruft `network_query_fn`; bei Attenuation: `sample_ct_volume` → μ; bei `emission=True`: `raw2outputs_emission` (inkl. optional raw Dump), sonst Standard-NeRF-Pfad.
  - `create_nerf(...)`: Baut NeRF-MLP(s) + Positional Encoding, setzt Output-Kanalzahl auf 1 für Emission, erzeugt `render_kwargs_train/test` inkl. Attenuation-Flags.
- `run_nerf_helpers_mod.py`: MLP und Ray-/Encoding-Helper.
  - `NeRF`: 8x256 ReLU-MLP, Skip bei Layer 4, kleiner positiver Output-Bias; liefert 1-Kanal-Emission (oder 4-Kanal-RGB).
  - `SineLayer`/`NeRF_Siren`: SIREN-Variante (derzeit ungenutzt).
  - `get_embedder`/`Embedder`: Sin/Cos-Positional-Encoding (max_freq_log2, num_freqs).
  - Ray-Utils: `get_rays` (perspektivische Rays aus Pose/focal), `get_rays_ortho` (orthografische parallele Rays, Gitter in Bildebene skaliert auf Größe), `ndc_rays` (Shift auf near-Plane, Normierung), `sample_pdf` (hierarchisches Resampling nach Gewichten).


Generator & Daten (`graf/`)
--------------------------
- `config.py`: Hilfen rund um YAML und Modellaufbau. `get_data(config)` lädt `SpectDataset`, ermittelt H/W aus echter AP-Projektion, berechnet formale focal aus FOV, radius aus Config, liefert `hwfr`. `build_models(config)` baut NeRF-Config (Chunkgrößen, z-Dimensionen, Emission + Attenuation), ruft `create_nerf`, setzt near/far, instanziiert Ray-Sampler (`FlexGridRaySampler`), Generator auf CUDA. `build_lr_scheduler(...)` liefert Step- oder MultiStep-LR-Scheduler. `save_config`/`update_config` speichern bzw. CLI-Overrides.
- `datasets.py`: `SpectDataset` liest `data/manifest.csv` (patient_id, ap_path, pa_path, ct_path, optional act_path). `_load_npy_image` lädt AP/PA, normalisiert pro Bild auf [0,1]; `_load_npy_ct` permutiert Volumen nach (D,H,W), skaliert; `_load_npy_act` permutiert + normalisiert. `__getitem__` gibt Dict mit Tensors (AP/PA/CT/ACT) und Meta.
- `generator.py`: Wrapper um NeRF-Rendering. H/W/focal/radius + Pixelkoordinaten, Trainings-Sampler (`ray_sampler`) und Full-Image-Sampler (`FullRaySampler`). `__call__(z, rays)` wählt Train-/Test-Render-Args, hängt z als `features` an, passt near/far bei Radius-Intervall an, ruft `render` und gibt flache Projektion (Eval: auch Disp/Acc). `sample_pose` toggelt feste AP/PA, sonst Zufalls-Pose; `sample_rays` nutzt Full-Bild bei Orthografie oder konfigurierten Sampler für Trainings-Patches. `render_from_pose` rendert komplettes Bild aus gegebener Pose (AP/PA), flatten, kann CT-Kontext anreichen. `set_fixed_ap_pa` definiert AP(+Z)/PA(-Z) Posen und orthografische Größe, `build_ct_context` formatiert CT [1,1,D,H,W], flipped x-Achse, speichert grid_radius für μ-Sampling.
- `transforms.py`: Ray-Sampling-Logik. `FlexGridRaySampler.sample_rays` erzeugt NxN-Grid in [-1,1], skaliert zufällig (annealbar) und verschiebt, liefert Grid für `grid_sample` → trainiert auf zufälligem Bildpatch; `FullRaySampler` gibt alle Pixelindizes zurück (volle Projektion); `ImgToPatch` zieht GT-Pixel passend zu gesampelten Rays; `RaySampler.__call__` baut Rays (perspektivisch oder orthografisch) und extrahiert gewünschte Pixel.
- `utils.py`: Kamera-/Sampling-Helfer (`look_at` berechnet Rotationsmatrix zum Ursprung, `sample_on_sphere`/`to_sphere` für Zufalls-Punkte – aktuell nicht genutzt).


Weitere Dateien
---------------
- `data/manifest.csv`: Manifest für Datensatzpfade (AP/PA/CT/optional ACT pro Fall).
- `results_spect/`: Standard-Ausgabepfad (Logs, Checkpoints, Previews).
- `nerf/__init__.py`, `graf/__init__.py`, `__init__.py`: Marker ohne Logik.
