FROM node:20-bookworm-slim

WORKDIR /app

COPY package*.json ./
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 make g++ \
 && ln -sf python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*
RUN npm ci

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]
