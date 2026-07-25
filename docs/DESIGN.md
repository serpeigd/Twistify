# Decisiones de diseño

Vivo. Se escribe según se decide, no al final. Un writeup reconstruido a
posteriori se nota: pierde los caminos descartados, que es lo interesante.

## D1 — Sin framework de orquestación

Ni LangChain, ni CrewAI, ni LangGraph. El pipeline son N pasos secuenciales
con una puerta de verificación. Una función lo expresa mejor y es depurable
con pdb.

LangGraph se justifica con estado cíclico y ramificación condicional. El único
punto de este sistema que lo tendrá es el retrieval de títulos oscuros
("no encuentro suficiente, reformulo y busco otra vez"). Cuando llegue ahí,
se reevalúa — y si se introduce, será solo en ese nodo.

Coste de la decisión: si el proyecto crece a bifurcaciones reales habrá que
reescribir orquestación. Asumido: es más barato que arrastrar la abstracción
desde el principio.

## D2 — El esquema permite estados inválidos

`Claim.source_id` es opcional aunque un hecho sin fuente sea inaceptable.

Si Pydantic lo rechazara al parsear, el fallo sería una excepción en vez de
una métrica. Y lo que se necesita saber no es "¿ha fallado?" sino "¿en qué
porcentaje de casos, en qué estrato, y cuánto lo reduce la intervención?".

La validación vive en el verificador, que es una etapa con nombre y número.

## D3 — El semáforo etiqueta el corpus, no el output

En el prompt original el modelo se autoetiquetaba 🟢/🟡/🔴. Autoevaluación:
el mismo componente que puede filtrar certifica que no ha filtrado.

Aquí el tier se asigna a los documentos recuperados, antes de generar. El
generador pre-visionado recibe solo GREEN. AMBER se trata como RED por
defecto (asimetría de coste: un bullet de menos vs. un spoiler publicado).

Esto convierte la seguridad en una propiedad del contexto en vez de en una
instrucción. Instrucción = el modelo puede desobedecer. Contexto = no puede
revelar lo que no tiene.

## D4 — Tres métricas o ninguna

`leakage` y `grounding` sin `richness` premian el silencio: un brief vacío
puntúa perfecto en ambas. Está cubierto por un test
(`test_empty_brief_scores_perfectly_on_safety`) para que no se pueda olvidar.

El trade-off seguridad/riqueza es el resultado del proyecto, no un detalle.

## D5 — El harness se prueba contra un generador falso

Con fugas plantadas y respuesta conocida. Si la métrica no reproduce el número
esperado, la métrica está rota. Se descubre antes de gastar en API.

Efecto lateral: los tests corren en CI sin secretos.

## D6 — El ground truth no se genera con un LLM

Sería medir la coherencia del modelo consigo mismo. El fallo exacto que el
proyecto existe para detectar. 15 min/título, a mano.

## Descartado

- **Multi-agente (researcher / writer / critic).** No hay decisión dinámica
  que delegar. Añade no-determinismo y coste a cambio de estética.
- **Markdown como salida del generador.** Es capa de render. El dato es JSON
  tipado, o el paso a TTS + montaje nunca llega.
- **Libros en la v1.** Retrieval distinto, sin equivalente a RT/Metacritic ni
  a datos de rodaje. Se implementa `SourceAdapter` para dos dominios, se
  implementa uno.

## Abierto

- ¿Umbral de AMBER? Depende de la calibración del juez.
- ¿Cuánto cuesta el juez por caso? len(superficie) x len(labels) llamadas.
  Es el componente más caro del pipeline. Medir antes de optimizar.
