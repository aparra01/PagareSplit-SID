"""Tests de validación de pagaré único (DocNative / PagareSplit)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.splitter import validar_pagare_pdf_sucursales

OP_A = "100001"
OP_B = "100002"


def _barcodes_from_raw(raw: dict[int, str]) -> list[list[dict]]:
    max_page = max(raw.keys(), default=0)
    pages: list[list[dict]] = [[] for _ in range(max_page)]
    for page, code in raw.items():
        pages[page - 1] = [{"texto": code, "formato": "CODE_39"}]
    return pages


@pytest.mark.parametrize(
    ("raw", "total", "expected_valid"),
    [
        ({1: OP_A}, 5, True),
        ({1: OP_A, 3: OP_A}, 4, True),
        ({1: OP_A}, 15, True),
    ],
)
def test_pagare_valido_un_codigo(raw: dict[int, str], total: int, expected_valid: bool) -> None:
    por_pagina = _barcodes_from_raw(raw)
    if len(por_pagina) < total:
        por_pagina.extend([[] for _ in range(total - len(por_pagina))])

    with patch("app.splitter.barcodes_pdf_en_memoria", return_value=por_pagina):
        result = validar_pagare_pdf_sucursales(pdf_bytes=b"%PDF", dpi=160)

    assert result["valido"] is expected_valid
    assert result["total_paginas"] == total


def test_multiples_pagares_distintos_barcode() -> None:
    por_pagina = _barcodes_from_raw({1: OP_A, 3: OP_B})
    por_pagina.extend([[]])

    with patch("app.splitter.barcodes_pdf_en_memoria", return_value=por_pagina):
        result = validar_pagare_pdf_sucursales(pdf_bytes=b"%PDF", validar_un_barcode=True)

    assert result["valido"] is False
    assert result["codigo"] == "MULTIPLES_PAGARES"


def test_validar_un_barcode_desactivado() -> None:
    por_pagina = _barcodes_from_raw({1: OP_A, 2: OP_B})

    with patch("app.splitter.barcodes_pdf_en_memoria", return_value=por_pagina):
        result = validar_pagare_pdf_sucursales(
            pdf_bytes=b"%PDF",
            validar_un_barcode=False,
        )

    assert result["valido"] is True


def test_repeticion_normal_dos_lecturas() -> None:
    por_pagina = _barcodes_from_raw({1: OP_A, 3: OP_A})
    por_pagina.append([])

    with patch("app.splitter.barcodes_pdf_en_memoria", return_value=por_pagina):
        result = validar_pagare_pdf_sucursales(
            pdf_bytes=b"%PDF",
            validar_repeticiones_barcode=True,
            max_repeticiones_barcode=2,
        )

    assert result["valido"] is True


def test_repeticion_excesiva_rechazo() -> None:
    por_pagina = _barcodes_from_raw({1: OP_A, 3: OP_A, 5: OP_A, 7: OP_A})
    por_pagina.extend([[], []])

    with patch("app.splitter.barcodes_pdf_en_memoria", return_value=por_pagina):
        result = validar_pagare_pdf_sucursales(
            pdf_bytes=b"%PDF",
            validar_repeticiones_barcode=True,
            max_repeticiones_barcode=2,
        )

    assert result["valido"] is False
    assert result["codigo"] == "BARCODE_REPETIDO_EXCESO"
    assert result["primera_pagina_excedente"] == 5


def test_repeticiones_desactivadas_pasa() -> None:
    por_pagina = _barcodes_from_raw({1: OP_A, 3: OP_A, 5: OP_A, 7: OP_A})
    por_pagina.extend([[], []])

    with patch("app.splitter.barcodes_pdf_en_memoria", return_value=por_pagina):
        result = validar_pagare_pdf_sucursales(
            pdf_bytes=b"%PDF",
            validar_repeticiones_barcode=False,
        )

    assert result["valido"] is True
