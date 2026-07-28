# Deploy em plataforma de nuvem

Este repositório pode ser implantado em uma plataforma que execute imagens
Docker, como Render ou outro provedor compatível. A plataforma deve construir o
`Dockerfile` da raiz e não deve substituir o `ENTRYPOINT` nem o `CMD` da imagem.

## Configuração do serviço

1. Crie um serviço web a partir deste repositório.
2. Selecione o runtime Docker e a branch que será implantada.
3. Não configure um comando de início personalizado. O `CMD` da imagem inicia o
   Uvicorn usando a variável `PORT`, com fallback local para `8000`.
4. Configure a verificação HTTP de saúde em `GET /healthz`.
5. Se o modelo precisar persistir entre implantações, monte um volume em
   `/data`.

## Variáveis de ambiente

Configure os valores adequados ao ambiente pela interface segura do provedor:

```dotenv
MEDTRACK_ENV=staging
MEDTRACK_LOG_LEVEL=INFO
MEDTRACK_MODEL_URI=/data/models/medtrack-yolo/v1.0.0/best.pt
MEDTRACK_MODEL_VERSION=v1.0.0
MEDTRACK_MODEL_MANIFEST=config/models/medtrack-yolo-v1.0.0.json
MEDTRACK_FETCH_MODEL_ON_START=true
MEDTRACK_DEVICE=cpu
MEDTRACK_MAX_IMAGE_DIMENSION=1024
MEDTRACK_YOLO_CONFIDENCE=0.5
EASYOCR_MODULE_PATH=/data/easyocr
MEDTRACK_CORS_ORIGINS=
```

Não copie um arquivo `.env` com segredos para o Git. A variável
`MEDTRACK_MODEL_URI` é obrigatória para o entrypoint preparar o diretório do
modelo antes de reduzir seus privilégios para o usuário `app`.

## Saúde e prontidão

- `GET /healthz` confirma que o processo HTTP está em execução e deve ser usado
  como healthcheck da plataforma.
- `GET /readyz` confirma que o modelo foi carregado e que o serviço está pronto
  para inferência.

Após o deploy, substitua `URL` pelo domínio fornecido pela plataforma:

```powershell
Invoke-WebRequest https://URL/healthz
Invoke-WebRequest https://URL/readyz
```

Os logs são enviados para stdout. Antes de promover uma versão, valide os dois
endpoints e confirme que o provedor preservou o volume do modelo.
