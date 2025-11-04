#!/usr/bin/env python3
"""
search_repo.py

Simple repository search tool: busca por palabras o expresiones regulares
en archivos de texto dentro de un directorio.

Uso básico:
    python search_repo.py "import pandas"  # busca en el directorio actual
    python search_repo.py -i -e .py,.ipynb "pandas" path/to/repo

Opciones:
  -i, --ignore-case     : búsqueda case-insensitive
  -r, --regex           : interpretar la consulta como expresión regular
  -e, --exts            : lista de extensiones separadas por coma (por ejemplo .py,.ipynb). Si no se indica, busca en todos los archivos de texto.
  -o, --json            : guardar salida en JSON en el archivo indicado

Salida: imprime por pantalla coincidencias: ruta, línea, número y snippet.
"""
import argparse
import os
import re
import json
from pathlib import Path


def is_likely_text(bytes_sample: bytes) -> bool:
    # Heurística: si contiene muchos bytes nulos u otros no imprimibles, lo tratamos como binario
    if not bytes_sample:
        return True
    nulls = bytes_sample.count(b"\x00")
    if nulls > 0:
        return False
    # proporción de bytes con valor < 9 (tab, lf, cr allowed)
    non_print = sum(1 for b in bytes_sample if b < 9 or (11 <= b <= 12) or (14 <= b <= 31))
    return (non_print / len(bytes_sample)) < 0.3


def search_in_file(path: Path, pattern, flags) -> list:
    matches = []
    try:
        with path.open("rb") as f:
            sample = f.read(4096)
            if not is_likely_text(sample):
                return matches
        # read as text, replace errors
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if pattern.search(line):
                    snippet = line.strip()
                    matches.append({"line": i, "snippet": snippet})
    except (OSError, UnicodeError):
        # omitir archivos que no se pueden leer
        return matches
    return matches


def compile_pattern(query: str, regex: bool, ignore_case: bool):
    flags = re.MULTILINE
    if ignore_case:
        flags |= re.IGNORECASE
    if regex:
        return re.compile(query, flags)
    # escape query for literal match
    return re.compile(re.escape(query), flags)


def walk_and_search(root: Path, pattern, exts=None, excludes=None) -> dict:
    results = {}
    excludes = excludes or set()
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories in-place so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for name in filenames:
            fp = Path(dirpath) / name
            if exts:
                if fp.suffix.lower() not in exts:
                    continue
            # skip very large files to avoid long blocking reads
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if size > walk_and_search.max_size_bytes:
                continue
            found = search_in_file(fp, pattern, None)
            if found:
                results[str(fp)] = found
    return results


def main():
    parser = argparse.ArgumentParser(description="Buscar texto/regex en un repositorio/directorio")
    parser.add_argument("query", help="Palabra o expresión a buscar")
    parser.add_argument("path", nargs="?", default=".", help="Directorio raíz (por defecto: .)")
    parser.add_argument("-i", "--ignore-case", action="store_true", help="Ignorar mayúsculas/minúsculas")
    parser.add_argument("-r", "--regex", action="store_true", help="Interpretar query como regex")
    parser.add_argument("-e", "--exts", help="Extensiones a incluir separadas por coma (ej: .py,.ipynb)")
    parser.add_argument("-o", "--json", help="Guardar salida en JSON en archivo dado")
    parser.add_argument("--max-size", type=float, default=5.0, help="Máximo tamaño de archivo en MB para escanear (por defecto: 5)")
    parser.add_argument("--exclude", help="Carpetas a excluir separadas por coma (nombres simples, ej: .git,venv,node_modules). Por defecto: .git,.venv,venv,__pycache__,node_modules,.ipynb_checkpoints")

    args = parser.parse_args()
    root = Path(args.path).resolve()
    exts = None
    if args.exts:
        exts = {e.strip().lower() if e.startswith('.') else '.' + e.strip().lower() for e in args.exts.split(',')}

    # configurar tamaño máximo (bytes) que se permite escanear por archivo
    walk_and_search.max_size_bytes = int(args.max_size * 1024 * 1024)

    # excluir carpetas por nombre (no rutas completas)
    default_excludes = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}
    excludes = default_excludes
    if args.exclude:
        extras = {s.strip() for s in args.exclude.split(",") if s.strip()}
        excludes = excludes.union(extras)

    pattern = compile_pattern(args.query, args.regex, args.ignore_case)
    results = walk_and_search(root, pattern, exts, excludes=excludes)

    # imprimir resultados legibles
    total = 0
    for fp, matches in results.items():
        print(f"File: {fp}")
        for m in matches:
            print(f"  {m['line']:5d}: {m['snippet']}")
            total += 1
        print()

    print(f"Total coincidencias: {total} en {len(results)} archivos")

    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as out:
                json.dump(results, out, ensure_ascii=False, indent=2)
            print(f"Salida JSON guardada en: {args.json}")
        except OSError as e:
            print(f"Error guardando JSON: {e}")


if __name__ == "__main__":
    main()
