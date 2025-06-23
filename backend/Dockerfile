# Stage 1: Use an official, lightweight Python image
# Using a specific version is better than 'latest' for reproducibility
FROM python:3.12.6

# Set environment variables for better Python performance in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first, to leverage Docker's caching mechanism
# This layer will only be re-built if requirements.txt changes
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the working directory
COPY . .

# Expose the port that the application will run on
EXPOSE 8000

# The command to run your application using uvicorn
# This tells uvicorn to run the 'app' object from the 'main.py' file.
# The host 0.0.0.0 is crucial for making the server accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]