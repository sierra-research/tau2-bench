#### Run Tau2 in docker containers

First change into the .docker directory then run these commands:

```
docker compose up -d
```

##### Download model from ollama

```
docker compose exec -ti ollama bash
ollama pull qwen3.6:27b
```

##### Bechmark model in tau2

```
docker compose exec -ti tau-bench bash
```

###### Create a .env file

```
echo "OLLAMA_API_BASE=http://ollama:11434" > .env
```

```
tau2 run --domain airline  --agent-llm ollama/qwen3.6:27b --user-llm ollama/qwen3.6:27b --num-trials 1 --num-tasks 1
```
