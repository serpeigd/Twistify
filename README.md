# 🎬 Twistify

**Antes de verla, sin spoilers. Después, con todos los giros.**

Twistify es una app de películas con una regla que se cumple de verdad, no
solo se promete: la trama nunca sale del servidor hasta que tú dices que ya
la has visto. Debajo, un harness de evaluación mide si esa promesa se
cumple — con números, no con un sello verde autoconcedido.

<p align="center">
  <img src="docs/screenshots/twistify-safe-mode.png" width="49%" alt="Modo sin spoilers">
  <img src="docs/screenshots/twistify-spoiler-mode.png" width="49%" alt="Modo spoiler abierto">
</p>

---

## Pruébalo en 2 minutos

```bash
git clone https://github.com/serpeigd/Twistify.git
cd Twistify
pip install fastapi "uvicorn[standard]" pydantic pyyaml
python webapp/app.py
# abre http://127.0.0.1:8000
```

Elige una película curada (Sixth Sense, Fight Club, Get Out, Parasite,
The Prestige, Se7en o Arrival), lee la ficha sin spoilers, y cuando quieras,
abre el telón.

## Qué hace distinto a esto de "otro CRUD con películas"

- **La partición de spoilers es una propiedad del servidor, no una promesa
  de la UI.** El contenido post-visionado no se manda al navegador hasta que
  el cliente declara `seen=true` — abrir devtools no revela nada. No es CSS
  escondiendo un `<div>`.
- **Cada afirmación factual dice si tiene fuente o no.** Sin inventar
  `source_id` falsos cuando no hay retrieval real. El hueco se muestra, no
  se disimula.
- **El propio detector de fugas de spoilers está medido, no asumido.** Hay
  un harness de evals (`evals/`) que calibra el juez contra fugas plantadas
  y reporta su recall real — incluyendo el caso incómodo en que el juez
  barato falla (ver [Resultados](#bajo-el-capó-el-harness-de-evaluación)).
- **Filtros con sentido.** Temáticas (identidad, obsesión, clase y poder…)
  que agrupan varias películas de verdad, no una etiqueta única por título.

## Cómo está hecho

| Pieza | Qué es |
|---|---|
| `webapp/` | FastAPI + vanilla JS. Sirve el catálogo, gestiona el gate de spoilers, comentarios (editar/borrar sin cuentas, con token anónimo por navegador). |
| `content/curated/*.json` | 7 fichas investigadas a mano (con fuentes citadas: Wikipedia, Hollywood Reporter, No Film School…), no generadas por un LLM sin verificar. |
| `src/preshow/` | Contratos de datos (Pydantic) tanto del contenido curado como del harness de medición. |
| `evals/` | El experimento real: métricas de fuga/fundamentación/riqueza, juez calibrado, dataset de 20 títulos estratificado. |

## Estado

| Qué | Estado |
|---|---|
| App Twistify (catálogo, gate de spoilers, filtros, comentarios) | ✅ 7/20 fichas curadas |
| Harness de evals offline | ✅ 8 tests en verde |
| Ground truth de spoilers (20 títulos) | ✅ 20/20, investigado con fuentes citadas |
| Generador baseline (sin retrieval) | ✅ código listo |
| Calibración del juez (offline) | ✅ recall=0.0 confirmado — justifica por qué hace falta un juez mejor |
| Medir el baseline sobre los 20 títulos | 🔴 pendiente de correr |
| Retrieval (TMDB/OMDb/Wikipedia) + verificador | ⬜ próximo hito |

## Bajo el capó: el harness de evaluación

La parte que no se ve en las capturas es la que sostiene la promesa de la
app: un sistema que **mide**, en vez de prometer, tres cosas por cada ficha:

1. **`leakage_rate`** — ¿se coló algún spoiler en el contenido pre-visionado?
2. **`grounded_fact_rate`** — ¿cuántas afirmaciones llevan fuente real?
3. **`richness`** — ¿cuánto dice realmente? (un output vacío puntúa perfecto
   en las dos primeras — por eso nunca se reporta sin esta)

```bash
python -m pytest tests/ -q                      # 8/8, sin red, sin API key
python evals/run_eval.py --generator baseline   # bloquea con <15 títulos etiquetados
```

Las decisiones detrás de este diseño (por qué no hay LangGraph, por qué el
esquema permite estados inválidos a propósito, por qué el ground truth no lo
puede generar el mismo modelo que se está midiendo) están documentadas en
[`docs/DESIGN.md`](docs/DESIGN.md).

## Fuentes y restricciones legales

- **TMDB** — gratis para uso no comercial, exige atribución. Sus términos
  restringen usar el contenido para *entrenar* sistemas de IA; inferencia con
  atribución es la lectura habitual, pero revísalo antes de escalar esto.
- **OMDb** — vía a puntuaciones de Rotten Tomatoes/Metascore, tier gratis
  limitado.
- **Wikipedia** — CC BY-SA, ya en uso para las fichas curadas.
- **Scraping de IMDb** — prohibido por ToS, no se hace bajo ninguna excusa.

## Licencia

Sin licencia definida todavía — repo de portfolio personal. Si quieres
reutilizar algo, pregunta antes.
