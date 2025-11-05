# Survey Automation Demo

JSON extraction from survey questionnaires using llama.cpp + LoRA.

## How to Run
1. Download the fine tuned gguf model from HF (https://huggingface.co/MRKumar/llama2_sp_automate/resolve/main/merged-q4-k-m.gguf) & set MODEL_PATH in script to point the gguf file.
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




## Debugging/Fixed issues:
1. LLM hallucinate and returns multiple JSON objects. Fixed with single JSON extract function.
2. For non questions - LLM sometime returns empty dict along with some explanations, breaking the JSON load. Fixed by passing a criteria for non questions in extract function.
3. Consider the non questions response of {} as valid, previously it was invalid.
