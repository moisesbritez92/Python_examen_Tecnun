# Buscador de texto para el repositorio

Este script `search_repo.py` permite buscar palabras o expresiones regulares en los archivos del repositorio.

Ejemplos (PowerShell / Windows):

- Buscar literal "import pandas" en el directorio actual (case-sensitive):

    python .\search_repo.py "import pandas"

- Buscar sin distinguir mayúsculas y filtrar solo archivos `.py` y `.ipynb`:

    python .\search_repo.py -i -e .py,.ipynb "pandas"

- Guardar resultado en JSON:

    python .\search_repo.py -i "pandas" -o resultados.json

Notas:
- El script intenta detectar archivos binarios y omitirlos.
- Para búsquedas avanzadas, usar `-r` para interpretar la consulta como regex.

Posibles mejoras futuras:
- Soporte para resaltar coincidencias en color.
- Interfaz web o integración con VSCode (extensión) para búsqueda rápida.
- Tests automáticos y métricas de rendimiento.

Comando rápido (buscar en el repositorio actual)
------------------------------------------------

Si quieres buscar rápido desde la raíz del repositorio (directorio actual), usa simplemente:

    python .\search_repo.py -i "TÉRMINO"

Ejemplo concreto (buscar 'import pandas' sin distinguir mayúsculas):

    python .\search_repo.py -i "import pandas"

    python .\search_repo.py -i "PCA" -o resultados_pca.json

Nota: no es necesario pasar la ruta si ya estás en la raíz del repo — el script usa `.` por defecto.
