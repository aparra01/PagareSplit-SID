"""Split de pagaré actual con dos Code39 iguales (pagaré pág. 1 + hoja de desembolso).

Formato Cooprogreso: el mismo No. Operación aparece como barcode en la hoja 1 del
pagaré y en la TABLA DE AMORTIZACIÓN Y COMPROBANTE DE DESEMBOLSO. A veces el cliente
entrega las hojas desordenadas (desembolso primero). En ambos casos debe salir un solo
pagaré con el código correcto.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.splitter import detectar_pagares_actual_por_barcode

OP_A = "0605129000"
OP_B = "0813601000"


def _por_pagina(total: int, raw: dict[int, str]) -> list[list[dict]]:
    pages: list[list[dict]] = [[] for _ in range(total)]
    for page, code in raw.items():
        pages[page - 1] = [{"texto": code, "formato": "CODE_39"}]
    return pages


def _detectar(total: int, raw: dict[int, str]) -> dict:
    # Layout sin portadas repetidas: fuerza la ruta por barcode Code39.
    scores = [{"page": p, "similarity": 1.0 if p == 1 else 0.0, "inicioLayout": p == 1} for p in range(1, total + 1)]
    with (
        patch("app.splitter._paginas_inicio_por_layout", return_value=([1], scores)),
        patch("app.splitter.barcodes_pdf_en_memoria", return_value=_por_pagina(total, raw)),
    ):
        return detectar_pagares_actual_por_barcode(pdf_bytes=b"%PDF", dpi=200)


def _rangos(result: dict) -> list[tuple[int, int, str]]:
    return [(p["pagina_inicio"], p["pagina_fin"], p["codigo_operacion"]) for p in result["pagares"]]


@pytest.mark.parametrize(
    ("caso", "raw", "esperado"),
    [
        # Pagaré (2 hojas) + desembolso (3 hojas): barcode en pág. 1 y pág. 3.
        ("ordenado", {1: OP_A, 3: OP_A}, [(1, 5, OP_A)]),
        # Desordenado: desembolso primero (pág. 1) y pagaré después (pág. 4).
        ("desordenado", {1: OP_A, 4: OP_A}, [(1, 5, OP_A)]),
        # Solo se leyó uno de los dos barcodes.
        ("un_solo_barcode_leido", {3: OP_A}, [(3, 5, OP_A)]),
    ],
)
def test_un_pagare_con_dos_barcodes_iguales_no_se_parte(caso, raw, esperado):
    result = _detectar(5, raw)
    assert _rangos(result) == esperado, caso


def test_lote_dos_pagares_uno_desordenado():
    # A ordenado (1-5, barcodes 1 y 3) + B desordenado (6-10: desembolso en 6, pagaré en 9).
    result = _detectar(10, {1: OP_A, 3: OP_A, 6: OP_B, 9: OP_B})
    assert _rangos(result) == [(1, 5, OP_A), (6, 10, OP_B)]
