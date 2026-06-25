"""Benchmark de detección CAPTURESEP en lotes de prueba."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import fitz
from capturesep_qr import detectar_marcadores_qr_capturesep, segmentos_entre_paginas_qr

PDFS = [
    Path(r"c:\Users\aparra\Downloads\Prueba1_0004 ACTA TRANSACCIONAL LOTE DE 2.pdf"),
    Path(r"c:\Users\aparra\Downloads\Prueba1_0005 - ATAS TRANSACCIONALES.pdf"),
    Path(r"c:\Users\aparra\Downloads\Prueba1_0006 - CENTROS DE MEDIACIÓN.pdf"),
    Path(r"c:\Users\aparra\Downloads\Prueba1_0007 PAGARES FORMATO ANTIGUO.pdf"),
    Path(r"c:\Users\aparra\Downloads\Prueba1_0008 PAGARE FORMATO ACTUAL.pdf"),
]


def bench(path: Path) -> dict | None:
    if not path.is_file():
        print(f"SKIP (no existe): {path.name}")
        return None
    pdf_bytes = path.read_bytes()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    doc.close()

    t0 = time.perf_counter()
    marcadores = detectar_marcadores_qr_capturesep(pdf_bytes=pdf_bytes)
    elapsed = time.perf_counter() - t0
    paginas_qr = [m["pagina_1_based"] for m in marcadores]
    segmentos = segmentos_entre_paginas_qr(total, paginas_qr)

    print(f"\n=== {path.name} ===")
    print(f"  paginas totales : {total}")
    print(f"  tiempo          : {elapsed:.2f}s")
    print(f"  marcadores QR   : {len(marcadores)} -> paginas {paginas_qr}")
    print(f"  documentos cola : {len(segmentos)}")
    for i, seg in enumerate(segmentos, 1):
        print(f"    doc {i}: {len(seg)} pags ({seg[0]}-{seg[-1]})")

    return {
        "name": path.name,
        "pages": total,
        "seconds": elapsed,
        "markers": len(marcadores),
        "docs": len(segmentos),
    }


if __name__ == "__main__":
    results = [r for p in PDFS if (r := bench(p)) is not None]
    if results:
        print("\n=== RESUMEN ===")
        print(f"{'Archivo':<50} {'Pags':>5} {'Tiempo':>8} {'QR':>4} {'Docs':>5}")
        print("-" * 75)
        for r in results:
            short = r["name"][:48] + (".." if len(r["name"]) > 50 else "")
            print(f"{short:<50} {r['pages']:>5} {r['seconds']:>7.2f}s {r['markers']:>4} {r['docs']:>5}")
        print(f"\nTotal: {sum(r['seconds'] for r in results):.2f}s en {len(results)} archivos")
