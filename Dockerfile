# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install system dependencies (if needed) and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=flask_api.app
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Command to run the Flask app
CMD ["flask", "run"]
