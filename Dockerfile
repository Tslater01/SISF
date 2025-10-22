# Dockerfile

# Stage 1: Use an official Python runtime as a parent image
FROM python:3.12-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy the files needed to install dependencies
# We copy these first to leverage Docker's layer caching.
COPY requirements.txt .
COPY pyproject.toml .

# Stage 4: Install the Python dependencies into the container
RUN pip install --no-cache-dir -r requirements.txt

# Stage 5: Install the project in editable mode
# This creates the link to our source code so Python can find the 'sisf' module.
COPY src/ ./src/
RUN pip install -e .

# Stage 6: Copy the rest of the application code
COPY .env .
COPY main_loop.py .
COPY oversight_app.py .

# Stage 7: Expose the ports the app and dashboard will run on
EXPOSE 8000
EXPOSE 8501

# Stage 8: Define the default command to run your API server
# This is the command that will be executed when the container starts.
CMD ["uvicorn", "sisf.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]