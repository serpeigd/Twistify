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
| 0 | Ground truth de spoilers (20 títulos) | 🔴 1/20 |
| 0 | Baseline sin retrieval + medición | 🔴 |
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

| Juez | precision | recall | n |
|---|---|---|---|
| Substring | — | — | — |
| LLM | — | — | — |

## Correr

```bash
pip install pydantic pytest pyyaml
python -m pytest tests/ -q      # no necesita red ni API key
python evals/run_eval.py --generator baseline
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
