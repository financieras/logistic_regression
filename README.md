Clonar el repositorio 'logistic_regression'

Crear un entorno virtual, por ejemplo, uno llamado 'env':

Ir a la carpeta del proyecto:

```bash
cd logistic_regression
```
Crear el entorno virtual con Python3:

```bash
python3 -m venv env
```

Activar el entorno virtual:

```bash
source env/bin/activate
```

Para aplicar el archivo requirements.txt en su entorno virtual activado en Ubuntu, siga estos pasos:

1. Asegúrese de que su entorno virtual 'env' esté activado. Debería ver (env) al inicio de su línea de comando.

2. Navegue al directorio donde se encuentra su archivo requirements.txt. Generalmente, este archivo está en la raíz del proyecto. Si ya está en ese directorio, puede omitir este paso.

3. Ejecute el siguiente comando para instalar todas las dependencias listadas en el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

Este comando le indicará a pip que instale todos los paquetes y sus versiones específicas tal como están listados en el archivo requirements.txt[1][3].

4. Espere a que pip complete la instalación de todas las dependencias. Verá una salida en la terminal que muestra el progreso de la instalación.

5. Una vez finalizada la instalación, puede verificar que los paquetes se hayan instalado correctamente usando:

```bash
pip list
```

Este comando mostrará todos los paquetes instalados en su entorno virtual.

Siguiendo estos pasos, habrá aplicado exitosamente el archivo requirements.txt a su entorno virtual en Ubuntu.