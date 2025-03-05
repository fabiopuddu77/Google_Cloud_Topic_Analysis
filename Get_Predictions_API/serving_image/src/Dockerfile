
# Use the official Python image
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu113.py310
 
# Set the working directory in the container
WORKDIR /
 
ENV HOST 0.0.0.0

# Copy the requirements file into the container
COPY requirements.txt .
 
# Install dependencies
RUN pip install -r requirements.txt
 
# Copy the rest of the application code into the container
COPY . /
 
# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]

EXPOSE 8081
