FROM python:3.11-slim

WORKDIR /project

ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev\  
    curl \
    libasound2-dev \
    libffi-dev \
    libportaudio2 \
    portaudio19-dev \
 && pip install --no-cache-dir pipx \
 && pipx install uv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN uv --version

COPY . .

RUN uv venv \
 && uv pip install . \
 && uv pip list

LABEL maintainer="Ali Elneklawy" \
      description="Versatile RAG system that powers real-time chat with data from the web and nearly all file formats, deployable as WhatsApp/Telegram bots or an iframe."

EXPOSE 80

CMD ["python", "src/main.py"]
