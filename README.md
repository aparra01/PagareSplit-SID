# PagareSplit-SID

Microservicio dedicado para separar lotes PDF de pagarés sin ejecutar OCR ni modelos ML.

## Objetivo

`PagareSplit-SID` detecta páginas de inicio de pagarés y devuelve rangos para que CaptureSoft pueda partir el PDF en documentos individuales. Corre separado de `lightgbm-SID`, por lo que la separación no se bloquea cuando OCR está procesando documentos.

## Endpoints

- `GET /health`
- `POST /detectar-pagares-actual`
- `POST /normalizar-pdf` — preprocesa escaneos (quita blancos, orientación 90°/180°/270°, deskew fino)

`POST /detectar-pagares-actual` recibe multipart:

- `file`: PDF.
- `dpi`: opcional, por defecto `120`.
- `solo_rangos`: opcional, `true` para separar rápido por rangos sin leer barcodes.
- `separar_qr`: opcional, `true` para detectar hojas marcadora **CAPTURESEP** (v1 pagarés o v2 módulo Separadores).
- `separar_barcode`: opcional, `true` por defecto. Con `separar_qr=true`, primero parte por QR y luego aplica barcode/layout en cada tramo.
- `eliminar_blancos`: opcional, por defecto `true` (env `PAGARE_SPLIT_ELIMINAR_BLANCOS`).
- `corregir_orientacion`: opcional, por defecto `true` — corrige hojas mal puestas en el escáner (OpenCV + heurísticas pagaré).
- `aplicar_deskew`: opcional, por defecto `true` — inclinación fina (~1°–15°).
- `aplicar_mejora_imagen`: opcional, por defecto `true` — mejora tipo PaperStream (fondo, crop, ruido, énfasis).
- `incluir_pdf_normalizado`: opcional, `false`. Si `true`, respuesta `multipart/mixed` con JSON + PDF normalizado.

`POST /normalizar-pdf` acepta los mismos flags de preproceso y devuelve solo el PDF corregido.

### Preproceso de escaneo (WIA / ADF)

Antes de layout y barcode, el servicio puede:

1. Descartar páginas en blanco (separadores).
2. Enderezar orientación gruesa (90° / 180° / 270°) con OpenCV y reglas de portada pagaré.
3. Aplicar deskew fino.
4. **Mejora de imagen** (réplica software de PaperStream IP): blanqueo de fondo, despeckle, rayas verticales, auto-crop, énfasis de texto.

La respuesta de detección puede incluir `paginas_blancas_excluidas` y `transformaciones` por página. Los rangos `pagares[].paginas` referencian el PDF **original** (sin índices de hojas blancas).

### Mejora nativa PaperStream vs réplica PagareSplit

| PaperStream IP (fi-7160 TWAIN) | PagareSplit (`scan_enhance.py`) |
|-------------------------------|----------------------------------|
| Auto rotation / deskew / crop | `scan_orientation.py` + `auto_crop_content` |
| Blank page skip | `is_blank_page_rgb` |
| Background cleanup / BGAdjust | `whiten_background` |
| Vertical streaks reduction | `reduce_vertical_streaks` |
| Image emphasis / OCR tuning | `emphasize_document` |
| iDTC binarización adaptativa | No (se mantiene color para barcode); opcional futuro |
| Punch hole / tab crop | No implementado |

**Nota:** con WIA el driver aplica menos procesamiento que con PaperStream TWAIN. PagareSplit compensa en post-proceso del PDF.

Variables en `.env` (prefijo `PAGARE_SPLIT_`): `ELIMINAR_BLANCOS`, `CORREGIR_ORIENTACION`, `APLICAR_DESKEW`, `PREPROCESS_DPI`, `BLANK_MEAN_THRESHOLD`, etc. Ver `.env.example`.

Respuesta principal:

```json
{
  "total_paginas": 17,
  "total_pagares": 3,
  "modo": "capturesep_qr+layout_portada_barcode_code39",
  "paginas_qr": [3, 10],
  "marcadores_qr": [
    { "pagina_1_based": 3, "raw": "CAPTURESEP|1|Pagarés Formato Actual|SEPARADOR_PAGARE", "payload": { "cliente": "...", "separador": "SEPARADOR_PAGARE", "version": 1 } }
  ],
  "pagares": [
    { "indice": 1, "pagina_inicio": 1, "pagina_fin": 2, "codigo_operacion": "0312212000", "paginas": [1, 2], "n_hojas": 2 }
  ]
}
```

### Hojas QR soportadas

| Origen | Payload ejemplo | Uso |
|--------|-----------------|-----|
| Formas → Descargar hoja separador QR | `CAPTURESEP\|1\|Cliente\|SEPARADOR_PAGARE` | Partir lote de pagarés al escanear |
| Configuraciones → Separadores → QR | `CAPTURESEP\|2\|Cliente\|87` | Partir por tipo/carpeta (id en BD) |

Imprima la hoja e insértela **entre** documentos antes de escanear. PagareSplit excluye la página QR del contenido y genera un documento por tramo.

**Rendimiento:** el escaneo QR usa tres vías en orden: (1) imagen embebida del QR en la hoja separadora (~15–25 ms), (2) recorte central (~5 ms/página), (3) tiers profundos solo en páginas que parecen marcadora. Las portadas de pagaré con mucho texto se **saltan** sin escanear. Un PDF que es **solo** la hoja QR devuelve `modo: capturesep_qr_solo_marcadores` al instante, sin pasar por barcode.

## Arranque Local

Por defecto escucha en `http://127.0.0.1:8006`.

### PowerShell

Instalar dependencias:

```powershell
cd "C:\Users\aparra\OneDrive - ECUACOPIA\Documentos\SID\PagareSplit-SID"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Levantar servicio:

```powershell
cd "C:\Users\aparra\OneDrive - ECUACOPIA\Documentos\SID\PagareSplit-SID"
.\.venv\Scripts\python.exe main.py
```

Verificar health:

```powershell
Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8006/health"
```

Probar separación:

```powershell
$pdf = "C:\RUTA\A\TU\ARCHIVO.pdf"

curl.exe -X POST "http://127.0.0.1:8006/detectar-pagares-actual" `
  -F "file=@$pdf;type=application/pdf" `
  -F "dpi=120" `
  -F "solo_rangos=true" `
  -F "separar_qr=true"
```

### CMD

Instalar dependencias:

```bat
cd /d "C:\Users\aparra\OneDrive - ECUACOPIA\Documentos\SID\PagareSplit-SID"
python -m venv .venv
".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Levantar servicio:

```bat
cd /d "C:\Users\aparra\OneDrive - ECUACOPIA\Documentos\SID\PagareSplit-SID"
".venv\Scripts\python.exe" main.py
```

Verificar health:

```bat
curl.exe http://127.0.0.1:8006/health
```

Probar separación:

```bat
set "PDF=C:\RUTA\A\TU\ARCHIVO.pdf"
curl.exe -X POST "http://127.0.0.1:8006/detectar-pagares-actual" -F "file=@%PDF%;type=application/pdf" -F "dpi=120" -F "solo_rangos=true"
```

### Bash

Instalar dependencias:

```bash
cd "/c/Users/aparra/OneDrive - ECUACOPIA/Documentos/SID/PagareSplit-SID"
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Levantar servicio:

```bash
cd "/c/Users/aparra/OneDrive - ECUACOPIA/Documentos/SID/PagareSplit-SID"
./.venv/Scripts/python.exe main.py
```

Verificar health:

```bash
curl http://127.0.0.1:8006/health
```

Probar separación:

```bash
PDF="/c/RUTA/A/TU/ARCHIVO.pdf"
curl -X POST "http://127.0.0.1:8006/detectar-pagares-actual" \
  -F "file=@${PDF};type=application/pdf" \
  -F "dpi=120" \
  -F "solo_rangos=true"
```

## Variables

- `PAGARE_SPLIT_HOST`: host FastAPI, defecto `0.0.0.0`.
- `PAGARE_SPLIT_PORT`: puerto, defecto `8006`.
- `PAGARE_SPLIT_MAX_PDF_MB`: tamaño máximo de PDF, defecto `150`.
- `PAGARE_SPLIT_DEFAULT_DPI`: DPI base para separación, defecto `120`.

## Integración

CaptureSoft debe apuntar a:

```env
PAGARE_SPLIT_URL=http://127.0.0.1:8006
```

`lightgbm-SID` queda dedicado a OCR/indexación en `8005`.
