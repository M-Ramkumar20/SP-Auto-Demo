# Survey Automation Demo

JSON extraction from survey questionnaires using llama.cpp + LoRA.

## How to Run
1. Download model from Gdrive (Link will be provided on request) & set MODEL_PATH in script to point the gguf file.
2. pip install llama-cpp-python streamlit langchain-community
3. `streamlit run Rapid_DEMO.py`
4. Upload .docx or use demo items
5. Download export.csv

## Config used
- Model: merged-q4-k-m.gguf (LoRA adapter merged)
- Temperature: 0.0
- Stop: "Input:"
- Max tokens: 400 (eval), auto-retry +500 if truncated
- Single-JSON guarantee: balanced-brace scan, rejects multi-object tail

## Safety
- Non-questions → {} (Valid% may show reduced value)
- Retry once on unbalanced braces

## Results
- Valid%: ~70% (incl. non-questions), 95% (excl. non-questions)
- Median latency: ~50s (CPU, no GPU)
- Stable outputs across runs (temp=0.0)

## Files
- Rapid_DEMO.py: main script
- Data_export.csv: simple export (input, valid, response)
- Output_df.csv: detailed metrics (latency, tokens, reason)
