FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train the model at build time so cold starts are instant
RUN python train_model.py

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
