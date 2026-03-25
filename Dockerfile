# Use a slim Python image to keep the size small
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV/Pillow if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for faster builds)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]