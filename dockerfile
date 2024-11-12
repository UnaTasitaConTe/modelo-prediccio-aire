FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar únicamente los archivos necesarios para la instalación de dependencias
COPY requirements.txt .

# Instalar las dependencias (usando la caché de Docker si no cambian)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Comando para iniciar la aplicación
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:3000", "app:app"]
