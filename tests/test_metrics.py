"""Tests del harness contra salidas con fugas PLANTADAS.

Sabemos la respuesta de antemano. Si la métrica no la reproduce, la métrica
está rota — y lo descubres ahora, no cuando lleves 200 llamadas a la API
gastadas y un número en el README que no significa nada.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import SubstringJudge, calibrate  # noqa: E402
from metrics import aggregate, evaluate_case, grounding  # noqa: E402
from preshow.generator import make_brief  # noqa: E402
from preshow.schemas import Claim, SpoilerLabel  # noqa: E402

LABEL = SpoilerLabel(
    id="x_dead",
    canonical="el protagonista lleva muerto toda la pelicula",
    paraphrases=["el protagonista no esta vivo", "uno de los dos es un fantasma"],
    severity="core",
)


def test_clean_brief_no_leak():
    b = make_brief("t1", [("Thriller psicologico ambientado en Filadelfia.", "wiki#1")])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert not r.leaked


def test_literal_leak_detected():
    b = make_brief("t1", [("Ojo: el protagonista lleva muerto toda la pelicula.", "wiki#1")])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert r.leaked and r.core_leaked
    assert r.leaks[0].where == "context_bullets[0]"


def test_leak_in_voiceover_is_counted():
    """Regresión: es fácil medir solo los bullets y olvidar el guion."""
    b = make_brief("t1", [("Nada sospechoso.", "wiki#1")],
                   voiceover="Uno de los dos es un fantasma.")
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert r.leaked
    assert "script" in r.leaks[0].where


def test_substring_judge_misses_paraphrase():
    """FALLO ESPERADO Y DOCUMENTADO.

    Este test no comprueba que el sistema funcione; comprueba que sabemos
    dónde NO funciona. Es la justificación cuantitativa del juez LLM.
    """
    b = make_brief("t1", [("La perspectiva desde la que lo ves es una mentira.", "w#1")])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert not r.leaked  # <- fuga real que el baseline no ve


def test_grounding_ignores_interpretations():
    claims = [
        Claim(text="Se rodo en Filadelfia.", kind="fact", source_id="w#3"),
        Claim(text="Se rodo en Madrid.", kind="fact", source_id=None),
        Claim(text="El color rojo marca lo sobrenatural.", kind="interpretation"),
    ]
    total, grounded = grounding(claims)
    assert (total, grounded) == (2, 1)


def test_empty_brief_scores_perfectly_on_safety():
    """El test más importante del fichero.

    Un brief vacío tiene 0% de fuga. Si tu README solo reporta leakage_rate,
    el sistema optimo es no decir nada. Por eso richness va SIEMPRE al lado.
    """
    b = make_brief("t1", [])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    agg = aggregate([r])
    assert agg["overall"]["leakage_rate"] == 0.0
    assert agg["overall"]["richness_claims_per_case"] == 0.0


def test_judge_calibration_reports_recall():
    data = [
        ("el protagonista no esta vivo", LABEL, True),   # TP
        ("una historia sobre el duelo", LABEL, False),   # TN
        ("nada de lo que ves es lo que parece", LABEL, True),  # FN
    ]
    cal = calibrate(SubstringJudge(), data)
    assert cal.recall == 0.5
    assert cal.precision == 1.0


def test_strata_are_reported_separately():
    leaky = make_brief("t1", [("el protagonista no esta vivo", "w#1")])
    clean = make_brief("t2", [("Un thriller de los noventa.", "w#2")])
    rs = [
        evaluate_case(leaky, [LABEL], "longtail", SubstringJudge()),
        evaluate_case(clean, [LABEL], "mainstream", SubstringJudge()),
    ]
    agg = aggregate(rs)
    assert agg["longtail"]["leakage_rate"] == 1.0
    assert agg["mainstream"]["leakage_rate"] == 0.0
