# ClinicalIE – Sistema de Extração de Informações de Textos Clínicos

Um sistema básico de **Information Extraction (IE)** voltado para textos clínicos, alinhado com competências de NLP, pré-processamento de dados e Machine Learning.

## Estrutura do Projeto

```
ClinicalIE/
├── config/
│   └── settings.yaml          # Configurações centralizadas
├── data/
│   ├── raw/                   # Textos clínicos brutos
│   ├── processed/             # Resultados processados
│   └── annotations/           # Anotações para treinamento
├── src/
│   ├── preprocessing/         # Limpeza, tokenização, normalização
│   ├── extraction/            # NER, regras, relações
│   ├── models/                # Treinamento e avaliação
│   ├── pipeline/              # Pipeline principal
│   ├── postprocessing/        # Entity Linking (esqueleto)
│   └── utils/                 # I/O e métricas
├── notebooks/                 # Análise exploratória
├── tests/                     # Testes unitários
├── main.py                    # Ponto de entrada
└── requirements.txt
```

## Como começar

1. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```

2. **Instalar modelo spaCy científico (recomendado)**
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
   ```
   Ou modelo genérico:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Executar exemplo**
   ```bash
   python main.py
   ```
   No Windows, se estiver usando o Python 3.12:
   ```powershell
   py -3.12 main.py
   ```

4. **Rodar testes**
   ```bash
   python -m pytest tests/ -v
   ```
   No Windows, se estiver usando o Python 3.12:
   ```powershell
   py -3.12 -m pytest tests/ -v
   ```

## Fine-tuning de NER com anotações próprias

Crie um arquivo JSONL, com um objeto por linha. `start` e `end` são offsets
de caracteres no texto original; `end` é exclusivo. Cada anotação deve incluir
`text` e `entities`, mesmo quando não houver entidades:

```json
{"text":"Paciente com IAM.","entities":[{"start":13,"end":16,"label":"DISEASE"}]}
{"text":"Sem sintomas no momento.","entities":[]}
```

O arquivo `data/annotations/train.jsonl` contém exemplos sintéticos para
demonstração. Revise e substitua/complete essas anotações com dados
representativos e validados por especialistas antes de qualquer uso clínico.
Treine em uma pasta nova para preservar o modelo existente:
O modelo base `models/clinical_ner` é baseado em `en_core_web_sm` (inglês);
com anotações sintéticas em português, este comando valida apenas o fluxo
técnico, não a adequação linguística.

```bash
python -m src.models.train_ner --annotations data/annotations/train.jsonl --base-model models/clinical_ner --output models/clinical_ner_demo --iterations 20
```

Avalie Precision, Recall e F1 em outro arquivo anotado, separado do treinamento:

```bash
python -m src.models.evaluate --model models/clinical_ner_demo --annotations data/annotations/evaluation_synthetic.jsonl
```

Cada execução gera também `reports/ner_evaluation_report.pdf` com métricas
gerais e por classe, contagens TP/FP/FN e os dados de identificação da avaliação.
Para escolher outro destino, use `--report caminho/do/relatorio.pdf`.

O avaliador agrega os resultados documento a documento, evitando misturar spans
com offsets iguais em textos diferentes, e mostra métricas micro e por classe.
Use correspondência parcial apenas como diagnóstico complementar:
`--match-type partial`. Para comparar modelos, mantenha o mesmo conjunto
independente; não use os dados de treino para reportar desempenho.
No conjunto sintético incluído (12 textos), a execução de referência do modelo
demonstrativo obteve Precision/Recall/F1 micro de 0,50/0,50/0,50; o modelo-base,
0/0/0. Esses números são apenas uma verificação reproduzível do exemplo
sintético, não uma estimativa de desempenho em dados clínicos reais.

Para iniciar um modelo spaCy vazio em vez de ajustar um pré-treinado, use
`--base-model blank:pt`. O treinamento verifica offsets inválidos, entidades
sobrepostas e alinhamento aos tokens; corrija esses dados antes de treinar. O
modelo final é salvo no diretório indicado por `--output`. O corpus incluído é
pequeno, sintético e não serve para medir desempenho real; o modelo resultante
não está validado para uso clínico. Para usá-lo no pipeline, ajuste
`nlp.model_name` em `config/settings.yaml` para o caminho salvo somente após
avaliação com um conjunto independente e representativo.

## Componentes principais

| Módulo | Descrição | Status |
|--------|-----------|--------|
| `ClinicalTextCleaner` | Limpeza e normalização de espaços | Pronto |
| `ClinicalNormalizer` | Expansão de abreviações médicas | Pronto (expansível) |
| `ClinicalNER` | Named Entity Recognition (spaCy/scispaCy) | Pronto |
| `RuleBasedExtractor` | Dosagens, PA, frequências por regex | Pronto |
| `RelationExtractor` | Relações clínicas por tipos e proximidade textual | Pronto |
| `EntityLinker` | Ligação a conceitos e sinônimos UMLS locais | Pronto (vocabulário limitado) |
| `train_ner.py` | Fine-tuning de NER spaCy | Pronto |
| `evaluate.py` | Avaliação NER micro e por classe | Pronto |

As extrações de relações e entity linking ficam habilitadas em
`config/settings.yaml`. As entidades retornadas incluem `linked` com CUI,
nome canônico, vocabulário e confiança; quando não há correspondência, o valor
é `null`. O entity linker usa um vocabulário local limitado (sem chamadas
externas ou credenciais) e pode ser ampliado em
`src/postprocessing/entity_linker.py`. As relações e entidades inferidas pelo
vocabulário clínico são heurísticas, não substituem validação clínica.

---