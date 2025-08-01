# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage layer caching
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . /app

# Expose the ports for the API (8000) and the UI (8501)
EXPOSE 8000
EXPOSE 8501

# Run the start script when the container launches
CMD ["./start.sh"]