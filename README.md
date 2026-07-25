# Pre-Show Reels

Genera material promocional pre-visionado (spoiler-free) y análisis
post-visionado a partir de una película, **con dos garantías medidas**:

1. **No filtra spoilers** — medido contra ground truth etiquetado a mano, con
   el detector calibrado contra un benchmark público.
2. **No inventa datos** — cada afirmación factual lleva fuente, o se descarta.

El entregable de este repo no son los guiones. Son los números.

---

## Estado

| Hito | Qué | Estado |
|---|---|---|
| — | Harness de evals (offline, determinista) | ✅ 8 tests en verde |
| — | Contrato de datos (Pydantic) | ✅ |
| — | Set de 20 títulos estratificado | ✅ |
| 0 | Ground truth de spoilers (20 títulos) | ✅ 20/20 (investigado por LLM con fuentes citadas, ver D7 en `docs/DESIGN.md` — no es etiquetado humano) |
| 0 | Baseline sin retrieval + medición | 🔴 código listo, falta correr con `ANTHROPIC_API_KEY` |
| — | UI de demo local (gratis, sin retrieval real) | ✅ `webapp/` |
| 1 | Retrieval + verificador | ⬜ |
| 2 | Partición de contexto por spoiler | ⬜ |
| 3 | Adaptador de libros | ⬜ |

## Resultados

Aún no hay. Se rellena cuando el Hito 0 esté medido.

| Config | leakage | core leakage | grounded facts | richness |
|---|---|---|---|---|
| Baseline (mainstream) | — | — | — | — |
| Baseline (cola larga) | — | — | — | — |

**Calibración del juez** (obligatorio antes de creerse la tabla de arriba):

| Juez | precision | recall | n | fuente |
|---|---|---|---|---|
| Substring | 0.0* | 0.0 | 183 | interna (`evals/calibrate_substring.py`), no benchmark externo |
| LLM | — | — | — | pendiente, requiere `ANTHROPIC_API_KEY` |

\* precision=0.0 aquí es "0/0 predicciones positivas", no "siempre se equivoca al decir sí" — el juez nunca dijo sí (tp=0, fp=0). El número real e importante es el recall=0.0: en un split held-out (mitad de las paráfrasis de cada spoiler como needles conocidas, la otra mitad como texto de prueba nunca visto), el `SubstringJudge` no detecta NINGUNA paráfrasis nueva del mismo spoiler. Confirma cuantitativamente el motivo de tener un juez LLM.

Calibración vs. benchmark externo (TV Tropes Movies, IMDB Spoiler Dataset): no se pudo hacer sin credencial — TV Tropes Movies (Boyd-Graber et al. 2013) no tiene descarga directa pública; el IMDB Spoiler Dataset vive en Kaggle y exige cuenta/API key del usuario.

## Correr

```bash
pip install pydantic pytest pyyaml anthropic
python -m pytest tests/ -q      # no necesita red ni API key
export ANTHROPIC_API_KEY=sk-...  # solo hace falta para --generator baseline
python evals/run_eval.py --generator baseline
```

### UI de demo (gratis, sin API key)

Muestra un brief por plantilla para cada uno de los 20 títulos y lo verifica
en vivo contra su ground truth real de spoilers, con el mismo harness que usa
`run_eval.py`. No genera contenido nuevo con un LLM — eso es lo único del
proyecto que cuesta crédito, y sigue sin ejecutarse (`baseline.py` está listo
pero no se ha corrido, ver tabla de Resultados arriba).

```bash
pip install fastapi "uvicorn[standard]"
python webapp/app.py
# abre http://127.0.0.1:8000
```

## Fuentes y restricciones legales

- **TMDB** — gratis para uso no comercial, exige atribución y logo. Ojo: sus
  términos se reservan el derecho a prohibir el uso de contenido TMDB *en
  conexión con o para entrenar* una aplicación basada en IA. Uso en inferencia
  con atribución es la interpretación habitual, pero **léelo antes de hacer
  esto público** y no lo uses para fine-tuning. TTL de caché no comercial: 6
  meses. Rate limit ~40 rps.
- **OMDb** — única vía limpia a puntuaciones de Rotten Tomatoes y Metascore.
  Tier gratis muy limitado en peticiones/día.
- **Wikipedia** — CC BY-SA. La sección `Plot` viene pre-marcada como
  spoiler por la propia estructura del artículo: regalo estructural.
- **Scraping de IMDb** — prohibido por sus ToS. No lo hagas y dilo en el
  writeup; saber dónde está la línea también es ingeniería.
- **Libros (Hito 3)** — la API de Goodreads se cerró en 2020 y LibraryThing
  también. Quedan Open Library y Hardcover (GraphQL). No hay equivalente a
  Rotten Tomatoes ni a "datos de rodaje": el esquema tendrá que degradar
  campos a opcionales por dominio.

## Benchmarks para calibrar el juez

- TV Tropes Movies (Boyd-Graber et al., 2013) — ~16k frases, ~50% spoiler.
- IMDB Spoiler Dataset (Misra, arXiv:2212.06034).
- Goodreads / UCSD Book Graph (Wan et al., 2019) — 1.3M reseñas, ~3% positivas.

Advertencia: clasifican "¿es spoiler?"; tú necesitas "¿revela ESTE spoiler?".
Es entailment, no clasificación. La calibración da una cota, no una validación.
