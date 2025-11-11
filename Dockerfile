# Base image: Use a lightweight Python base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PORT=10000

# Install dependencies
WORKDIR /app
COPY requirements.txt /app/

# Install the dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py /app/

# Expose the correct port
EXPOSE 10000

# Run the uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]