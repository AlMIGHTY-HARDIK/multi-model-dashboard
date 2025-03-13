FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY multi-model-dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY multi-model-dashboard/ .

# Expose the port your app listens on (adjust if needed)
EXPOSE 8050

# Start the app using python3 instead of python
CMD ["python3", "app.py"]
