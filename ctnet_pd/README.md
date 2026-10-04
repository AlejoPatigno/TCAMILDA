# ctnet_pd — código de los experimentos

Paquete que implementa la metodología de `paper/materials_methods_v2.md`. Todos los hiperparámetros están en `configs/default.yaml`; el código no tiene constantes ocultas.

```
ctnet_pd/
├── configs/default.yaml        # todos los hiperparámetros (preprocesamiento, modelo, CV, búsqueda, XAI, estadística)
├── notebooks/
│   ├── 01_ctnet_neurovoz_experiments.ipynb   # notebook consolidado (Kaggle)
│   └── legacy/ctnet-neurovoz_original.ipynb  # notebook del estudio preliminar, sin cambios
├── scripts/run_experiment.py   # CLI: experimentos within / external
├── src/ctnet_pd/
│   ├── config.py               # carga, overrides "seccion.clave=valor", validaciones
│   ├── data.py                 # §2.2–2.3  indexación, familias de tareas, metadatos, Tabla 1
│   ├── features.py             # §2.5      log-Mel, operador R (ventanas), normalización por banda
│   ├── splits.py               # §2.4      folds repetidos disjuntos por sujeto, split interno
│   ├── model.py                # §2.6      CTNet (stem -> Transformer -> GAP + lineal)
│   ├── training.py             # §2.10.3, §2.14  pesos por sujeto, entrenamiento, KerasTuner, entrenador con prior
│   ├── metrics.py              # §2.7, §2.15  agregación por sujeto, métricas, calibración, IC bootstrap
│   ├── stats.py                # §2.16     DeLong pareado, bootstrap pareado, Wilcoxon, Holm
│   ├── xai.py                  # §2.8–2.9  q_pre (Grad-CAM), q_post (contribución exacta), q_J, Score-CAM, saliency
│   ├── grid.py                 #           rejilla 14×25 <-> 128×229 (D_RF y upsampling)
│   ├── faithfulness.py         # §2.11.1–2.11.4  borrado/inserción, máscaras aleatorias emparejadas, ROAR
│   ├── sanity.py               # §2.11.5   aleatorización en cascada y de etiquetas
│   ├── reproducibility.py      # §2.12     perfiles de frecuencia, nulos por permutación, techo de ruido
│   ├── priors.py               # §2.10     mapas de conceptos acústicos, prior, control aleatorio
│   ├── baselines.py            # Exp. II   SVM/eGeMAPS, sonda sobre embeddings congelados (WavLM, XLS-R, AST)
│   ├── pipeline.py             # Exp. I–IV CV anidada (reanudable) y validación externa
│   └── analysis.py             #           análisis sobre modelos guardados: mapas, fidelidad, prueba H1
└── tests/                      # 30 pruebas, incluida una de punta a punta con audio sintético
```

## Uso en Kaggle

1. Clona el repositorio (la sesión necesita internet) o sube la carpeta `ctnet_pd` como Dataset.
2. Abre `notebooks/01_ctnet_neurovoz_experiments.ipynb`, ajusta `OVERRIDES` y ejecuta. Antes de la corrida completa, revisa la tabla de familias de tareas y la prueba rápida de la sección 4.

Desde la línea de comandos:

```bash
python scripts/run_experiment.py within --cohort neurovoz --task-family ddk_pataka
python scripts/run_experiment.py within --cohort neurovoz --variant cnn_only --set model.n_transformer_layers=0
python scripts/run_experiment.py external --source pcgita --target neurovoz --task-family monologue
```

Los resultados quedan en `outputs/<cohorte>/<familia>/<variante>/` (`folds.csv`, `predictions.csv` por segmento y `runs.jsonl` con los hiperparámetros elegidos y la mejor época de cada fold). Si la corrida se interrumpe, al relanzarla se saltan los folds ya terminados.

## Pruebas

```bash
pip install -e ".[dev,notebook]"
pytest -q
```

## Pendientes

- PC-GITA: configurar `cohorts.pcgita` (ruta y `filename_regex`).
- Comparadores AttnLRP y *attention rollout* (sección 2.9).
- Congelar los conceptos del prior (`prior.concepts`) antes del Experimento IV.
