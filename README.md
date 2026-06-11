# PagareSplit-SID

Microservicio dedicado para separar lotes PDF de pagarés sin ejecutar OCR ni modelos ML.

## Objetivo

`PagareSplit-SID` detecta páginas de inicio de pagarés y devuelve rangos para que CaptureSoft pueda partir el PDF en documentos individuales. Corre separado de `lightgbm-SID`, por lo que la separación no se bloquea cuando OCR está procesando documentos.

## Endpoints

- `GET /health`
- `POST /detectar-pagares-actual`

`POST /detectar-pagares-actual` recibe multipart:

- `file`: PDF.
- `dpi`: opcional, por defecto `120`.
- `solo_rangos`: opcional, `true` para separar rápido por rangos sin leer barcodes.

Respuesta principal:

```json
{
  "total_paginas": 17,
  "total_pagares": 3,
  "modo": "layout_portada_barcode_code39",
  "pagares": [
    { "indice": 1, "pagina_inicio": 1, "pagina_fin": 7, "codigo_operacion": "0312212000", "paginas": [1, 2, 3, 4, 5, 6, 7], "n_hojas": 7 }
  ]
}
```

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
  -F "solo_rangos=true"
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
