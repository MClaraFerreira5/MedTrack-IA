## Objetivo

<!-- Explique o problema e o resultado deste PR. -->

Closes #

## Escopo

<!-- Liste as mudanças incluídas e o que ficou explicitamente fora. -->

## Evidências

<!-- Inclua comandos, resultados, screenshots ou vídeos relevantes. Não inclua dados sensíveis. -->

## Riscos e rollback

<!-- Descreva impacto na API, modelo, dados, contêiner ou deploy e como reverter. -->

## Checklist

- [ ] O PR tem escopo único e a branch está atualizada com a `main`.
- [ ] Os commits seguem Conventional Commits.
- [ ] Executei lint, checagem de tipos e testes automatizados.
- [ ] Verifiquei o lockfile com `uv lock --check`.
- [ ] Executei `git diff --check`.
- [ ] Adicionei ou atualizei testes para o comportamento alterado.
- [ ] A cobertura dos pacotes tocados não foi reduzida sem justificativa.
- [ ] Validei o build do contêiner, quando a mudança o afeta.
- [ ] Não incluí segredos, pesos, datasets, credenciais ou dados reais de pacientes.
- [ ] Revisei logs, artefatos, imagens e payloads quanto a dados pessoais ou de saúde.
- [ ] Mudanças no modelo preservam manifesto, versionamento, checksum e rollback.
- [ ] Mudanças no contrato HTTP atualizaram `docs/contracts/api-v1.md` e seus consumidores.
- [ ] Atualizei documentação/ADR quando alterei uma decisão técnica.
- [ ] Documentei rollback para mudanças incompatíveis, de configuração ou release.
- [ ] Não misturei upgrade amplo de dependências com refatoração funcional.

## Testes executados

```text
uv lock --check
uv run --group dev ruff check src tests scripts
uv run --group dev pyright src/medtrack_ai tests scripts
uv run --group dev pytest
docker compose build
git diff --check
```
