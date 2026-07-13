FROM python:3.10-slim

# Install system dependencies (curl + unzip + ca-certificates for Bun)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Bun runtime
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

# Create a system-level python alias for safety
RUN ln -sf "$(which python3)" /usr/local/bin/python

WORKDIR /app

# Step 1: Copy and install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Copy and install Bun dependencies (cached layer)
COPY backend/package.json backend/bun.lock* backend/tsconfig.json ./backend/
RUN cd backend && bun install

# Step 3: Copy the rest of the application files
COPY . .

# Env override for Hono server python child process execution path
ENV PYTHON_BIN=/usr/local/bin/python
EXPOSE 8080

CMD ["sh", "-c", "cd backend && bun run start"]
