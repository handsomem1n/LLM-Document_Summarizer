# Use the official Python image as the base image
FROM python:3.9

# Set the working directory
WORKDIR /

EXPOSE 8000

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y libmagic-dev
RUN pip install -r requirements.txt

# Copy the app code into the container
COPY . .

# Run the FastAPI server with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
