# Use an official Python image with Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies first
COPY multi-model-dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY multi-model-dashboard/ .

# Expose the port (if your app runs on 8050)
EXPOSE 8050

# Start the app
CMD ["python", "app.py"]
