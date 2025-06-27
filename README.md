# 🤖 Clasificador de Anuncios con IA

Este proyecto es una aplicación web interactiva que utiliza modelos de lenguaje grandes (LLMs) para clasificar textos e imágenes, determinando si son anuncios o no. Permite configurar diferentes modelos (Google Gemini, OpenAI GPT, o modelos locales a través de LM Studio) y gestionar ejemplos para mejorar la precisión del clasificador.

## ✨ Características

*   **Clasificación de Texto:** Introduce cualquier texto y el modelo lo clasificará como "anuncio" o "no_anuncio", proporcionando una breve explicación.
*   **Procesamiento de Imágenes:** Sube una imagen y un modelo de visión generará una descripción detallada, que luego podrás usar para clasificarla como anuncio o no.
*   **Configuración de Modelos:** Cambia fácilmente entre diferentes proveedores de LLMs (Google API, OpenAI API, LM Studio) y configura sus credenciales o detalles de conexión directamente desde la interfaz web.
*   **Gestión de Ejemplos:** Añade, edita y elimina ejemplos de clasificación para entrenar y mejorar el rendimiento del modelo.
*   **Interfaz Intuitiva:** Diseño limpio y moderno basado en Bootstrap para una experiencia de usuario agradable.

## 🚀 Tecnologías Utilizadas

*   **Backend:** Python 3.x con Flask
*   **LLM Framework:** DSPy
*   **APIs de LLM:**
    *   Google Generative AI (para modelos Gemini)
    *   OpenAI (para modelos GPT)
    *   LM Studio (para modelos locales que emulan la API de OpenAI)
*   **Frontend:** HTML, CSS (Bootstrap 5), JavaScript
*   **Manejo de Imágenes:** Pillow (PIL)

## 📋 Prerrequisitos

Antes de empezar, asegúrate de tener instalado lo siguiente:

*   **Python 3.11+**
*   **pip** (gestor de paquetes de Python)
*   **Homebrew** (recomendado para macOS para instalar `gcloud CLI`)
*   **Google Cloud CLI (`gcloud`)** (Opcional): Solo necesario si planeas usar la autenticación de Google Cloud (Application Default Credentials) en lugar de una API Key directa para la Google API. Si solo usas una API Key, puedes omitir este paso.
    *   Instálalo con Homebrew: `brew install --cask google-cloud-sdk`
    *   O sigue las instrucciones oficiales: [Instalar gcloud CLI](https://cloud.google.com/sdk/docs/install)
*   **LM Studio** (Opcional, si deseas usar modelos locales): Descárgalo desde [LM Studio](https://lmstudio.ai/). Asegúrate de cargar un modelo de visión compatible (ej. LLaVA, Gemma-3-4b) y de que su servidor esté corriendo en la IP y puerto configurados.

## ⚙️ Instalación y Configuración

Sigue estos pasos para poner el proyecto en marcha:

1.  **Clona el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd clasificador_anuncios
    ```

2.  **Crea y activa un entorno virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instala las dependencias de Python:**
    ```bash
    pip install -r requirements.txt
    pip install Pillow google-generativeai openai # Asegúrate de que estas estén instaladas
    ```
    *(Nota: `requirements.txt` debería contener la mayoría, pero estas últimas son cruciales para la funcionalidad de imagen y pueden no estar siempre en el `requirements.txt` inicial.)*

4.  **Configura tus API Keys (si usas Google o OpenAI):**
    *   La aplicación gestiona las API Keys a través de la interfaz web en la sección "Configuración del Modelo".
    *   **Para Google API:** Necesitarás una API Key de Google Generative AI. Puedes obtenerla en [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   **Para OpenAI API:** Necesitarás una API Key de OpenAI. Puedes obtenerla en [OpenAI API Keys](https://platform.openai.com/api-keys).
    *   **Importante:** Nunca subas tus API Keys directamente al código fuente en GitHub. Este proyecto las gestiona a través de un archivo `config.json` que se crea localmente.

5.  **Configura LM Studio (si usas modelos locales):**
    *   Abre LM Studio y descarga un modelo de visión (ej. `google/gemma-3-4b` o un modelo LLaVA).
    *   Inicia el servidor del modelo en LM Studio. Anota la IP y el puerto (por defecto suele ser `http://localhost:1234`).
    *   En la interfaz web de la aplicación, selecciona "LM Studio (Local)" y configura la IP y el Puerto.

## ▶️ Ejecutar la Aplicación

Para iniciar el servidor Flask, asegúrate de estar en el directorio `clasificador_anuncios/` y ejecuta:

```bash
source venv/bin/activate # Si no está activado
python app.py
```

El servidor se iniciará en `http://127.0.0.1:5000`. Abre esta URL en tu navegador web.

## 💡 Uso

Una vez que la aplicación esté corriendo:

*   **Clasificar Texto:** Escribe un texto en el campo principal y haz clic en "Clasificar".
*   **Procesar Imagen:** Sube una imagen en la sección "Procesar Imagen". Haz clic en "Procesar Imagen" para que el modelo genere una descripción. Esta descripción se cargará en el campo de texto principal, lista para ser clasificada.
*   **Configuración del Modelo:** En la columna de la derecha, selecciona el tipo de modelo que deseas usar (LM Studio, OpenAI, Google) y proporciona las credenciales o detalles de conexión necesarios. Haz clic en "Guardar Configuración".
*   **Gestión de Ejemplos:** En la sección inferior, puedes añadir nuevos ejemplos para mejorar el modelo. Haz clic en "Ver Ejemplos Existentes" para expandir la lista y usar los botones de "Acciones" para editar o eliminar ejemplos.

## ⚠️ Solución de Problemas Comunes

*   **`Address already in use`**: Si el puerto 5000 ya está en uso, significa que una instancia anterior del servidor no se cerró correctamente.
    1.  En tu terminal, busca el proceso que usa el puerto: `lsof -i :5000`
    2.  Termina el proceso usando su PID (reemplaza `PID_DEL_PROCESO`): `kill -9 PID_DEL_PROCESO`
    3.  Vuelve a iniciar la aplicación.
*   **`API Key no configurada`**: Asegúrate de haber introducido tu API Key en la interfaz web y haber guardado la configuración.
*   **`Model "..." not found` (LM Studio)**: Verifica que el nombre del modelo en la configuración de la aplicación (`config.json` o la interfaz web) coincida exactamente con el nombre del modelo que tienes cargado y ejecutándose en LM Studio.
*   **`Error processing image: ...`**:
    *   Asegúrate de que el modelo seleccionado (Google, OpenAI, LM Studio) sea un modelo de visión.
    *   Verifica que tu API Key sea correcta y tenga los permisos necesarios.
    *   Para LM Studio, confirma que el servidor del modelo esté activo y accesible en la IP/Puerto configurados.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto, por favor, abre un "issue" o envía un "pull request".

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
