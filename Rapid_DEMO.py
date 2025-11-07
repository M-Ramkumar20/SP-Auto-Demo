import pandas as pd
import os, time, json, re
import streamlit as st
from llama_cpp import Llama
from langchain_community.document_loaders import Docx2txtLoader
from io import StringIO
import tempfile
from datetime import datetime


# ------------------------------- Already merged the base with adapters and converted to gguf.  --------------------------------
## CONFIG USED
# BASE_MODEL_ID = 7b-chat-hf_Path
# ADAPTER_ID    = "MRKumar/SP_Automate-2025_10_22_16.12.34"
# OUT_DIR       = "/kaggle/working/merged-lora"

############## ALREADY PROCESSED.
# 1.
# !git clone https://github.com/ggerganov/llama.cpp
# %cd llama.cpp

# 2.
# !pip install -r requirements.txt

# 3.
# !python convert_hf_to_gguf.py ../../Artifacts/FineTuned_Model_with_LoRA-V1 --outfile ../../Artifacts/FT_Model_with_LoRA-V1-Quant/merged_full_precision.gguf

# 4. Requires cmake to be installed to run quantize.exe.
# .\build\bin\Release\llama-quantize.exe ..\..\Artifacts\FT_Model_with_LoRA-V1-Quant\merged_full_precision.gguf ..\..\Artifacts\FT_Model_with_LoRA-V1-Quant\merged-q4-k-m.gguf q4_k_m

# ------------------------------- END  --------------------------------



## Make it True to print the debugging statements.
debug = False


# ------------------------------- Config --------------------------------
# MODEL_PATH = PASS THE GGUF PATH.
MODEL_PATH = "../Artifacts/FT_Model_with_LoRA-V1-Quant/merged-q4-k-m.gguf"
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.0
REQUIRED_KEYS = [
    "question_type", "question_id", "question_title", "question_text",
    "entry_condition", "programming_instruction", "min", "max", "response_options"]
RUNS_CSV = "./runs.csv"
VALID_WARN_THRESHOLD = 95.0

APP_PASSWORD = os.getenv("APP_PASSWORD", "1234")

few_shot_examples = [
{"input": "S12) BRAND AWARENESS\n Before today, were you aware of the brand [BRAND NAME]?\n Yes\n No\n", "output": {"question_type": "Single choice", "question_id": "S12", "question_title": "BRAND AWARENESS", "question_text": "Before today, were you aware of the brand [BRAND NAME]?", "entry_condition": "", "programming_instruction": "", "min": "", "max": "", "response_options":  [{ "response_text": "Yes", "response_id": "1", "anchor": "False", "other_specify": "False", "mutually_exclusive": "False", "terminate": "False" }, { "response_text": "No", "response_id": "2", "anchor": "False", "other_specify": "False", "mutually_exclusive": "False", "terminate": "False" } ]}},
{"input": "The PC team will be launching a new brand. After several rounds of name ideations and preliminary testing the team would like to test up to 23 names with target consumers\n", "output": {}},
{"input": "\nWe are looking to talk to people about some products\n", "output": {}},
{"input": "\n60-40 split, create variable", "output": {}},
{"input": "\nZ4) Price\nWhat is the cost of the gadget?\nUSD 10-100 \n", "output": {"question_type": "Open numeric", "question_id": "Z4", "question_title": "Price", "question_text": "What is the cost of the gadget?", "entry_condition": "", "programming_instruction": "", "min": "10", "max": "100", "response_options":  []}},
{"input": "Create Fountain Soda variable if select anything in green. \nNatural fallout monitor for readable sample.\n", "output": {}},
{"input": "\nX1) OORU\nWhere do you reside?\n", "output": {"question_type": "Single choice", "question_id": "X1", "question_title": "OORU", "question_text": "Where do you reside?", "entry_condition": "", "programming_instruction": "", "min": "", "max": "", "response_options":  []}},
{"input": "Y1) OLOC\nWhere is your office located?\n", "output": {"question_type": "Single choice", "question_id": "Y1", "question_title": "OLOC", "question_text": "Where is your office located?", "entry_condition": "", "programming_instruction": "", "min": "", "max": "", "response_options":  []}},
{"input": "Ask only if selected 'Definitely would switch' in C1.\nC2) PRODUCT EXPERIENCE\n What's the main reason you prefer sugar-free drinks?\n Health concern\n Taste preference\n Calorie control\n Other\n", "output": {"question_type": "Single choice", "question_id": "C2", "question_title": "PRODUCT EXPERIENCE", "question_text": "What's the main reason you prefer sugar-free drinks?", "entry_condition": "Ask only if selected 'Definitely would switch' in C1", "programming_instruction": "", "min": "", "max": "", "response_options":  [{ "response_text": "Health concern", "response_id": "1", "anchor": "False", "other_specify": "False", "mutually_exclusive": "False", "terminate": "False" }, { "response_text": "Taste preference", "response_id": "2", "anchor": "False", "other_specify": "False", "mutually_exclusive": "False", "terminate": "False" }, { "response_text": "Calorie control", "response_id": "3", "anchor": "False", "other_specify": "False", "mutually_exclusive": "False", "terminate": "False" }, { "response_text": "Other", "response_id": "4", "anchor": "False", "other_specify": "False", "mutually_exclusive": "False", "terminate": "False" } ]}},
]


# -------- Helpers
def normalize_text(txt: str) -> str:
    t = txt.strip()

    # replace ' & " between letters (like Sam's) with ’ &  (safe apostrophe)
    t = re.sub(r"(?:(?<=\w)|(?<=\]))'(?=\w)", "’", t)
    t = re.sub(r'(?:(?<=\w)|(?<=\])|(?<=\s))"((?=\w)|(?=\s))', "’", t)

    # using re replace ' with " only for keys and values. 
    t = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r'"\1":', t) # keys
    t = re.sub(r":\s*'([^']*)'\s*(?=[,}])", r': "\1"', t) # values

    # Replace safe aphostrophe back to single quotes.
    t = re.sub(r"(?:(?<=\w)|(?<=\])|(?<=\s))’((?=\w)|(?=\s))", "'", t)

    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1]
    return t

def build_prompt(item: dict) -> str:
    lines = """
        Extract structured JSON only from valid survey questions. if input is not a valid question then return {}.
        Output only valid JSON (no extra text) or {}.
        JSON must start with { and end with }.
        Use double quotes for all keys and values.
        If the Input is not a question, just instructions, targets, timelines, quotas, or general info then give the output as {}
        Use exactly these keys, if any of the keys are missing in the input then return "" or [] or {} dont add anything on your own.
    """
    lines = lines + ",".join(REQUIRED_KEYS) + ".\n"
    prompt = []
    for shots in few_shot_examples:
        prompt.append(f"\nInput: \n{shots['input']}\nOutput (JSON): \n{shots['output']}")
    prompt.append("\nInput: " + item['text'] + "\n\nOutput (JSON): \n")
    return lines + "\n".join(prompt)

def parse_qnr(path):
    loader = Docx2txtLoader(tmp_file_path)
    data = loader.load()
    docs = data[0].page_content

    items = []
    chunks = docs.split("#QuestionBreak#")
    
    for chunk in chunks:

        chunk = re.sub(r"[\t]+", "", chunk)  ## replace tab chars
        chunk = re.sub(r"\n{2,}", "\n", chunk) ## replace only 2+ new lines chars with 1.
        qbreak = {"text" : chunk}
        items.append(qbreak)
    return items

## Extract only 1 json, simple regex not giving correct results.
def extract_single_json_balanced(t: str) -> tuple[dict,str]:
    t = t.strip()
    start = t.find("{")
    if start < 0: 
        return {}, "no_braces"
    depth, end = 0, -1
    i = start
    while i < len(t):
        c = t[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end < 0:
        return {}, "unbalanced_braces"
    core = t[start:end+1]
    if debug:
        print("\nCore is:", core)

    if end == 1:
        return core, "Non question"
    try:
        obj = json.loads(core)
    except Exception:
        return {}, "json_parse_error"
    tail = t[end+1:].lstrip()
    if tail.startswith("{"):
        return {}, "multi_obj_tail"
    return core, ""

def v_required(obj: str, req: list[str]) -> tuple[bool,str]:
    if (obj == "{}"):
        return True, "non_question"

    obj = json.loads(obj)
    if debug:
        print("v_required obj is: ", obj)
        print("v_required type is: ", type(obj))

    if not isinstance(obj, dict):
        return False, "not_dict"
    miss = [k for k in req if k not in obj]
    return (len(miss)==0, "" if not miss else f"missing:{','.join(miss)}")


def auto_repair_once(text: str) -> tuple[str, list[str]]:
    t = normalize_text(text)
    # trim to balanced object
    obj, err = extract_single_json_balanced(t)
    # print("auto func extract err is : ",err)
    if err == "":
        # return t, ""
        return obj, ""
    if obj == "{}":
        return obj, ["non_question"]
    return t, ["quotes_norm", err]


# ------------------------------------------------------------------------

@st.cache_resource
def load_llm(ctx):
    llm = Llama(model_path=MODEL_PATH, n_ctx=ctx, n_threads=0, verbose=0)
    return llm


def run_item(llm, item:dict):
    prompt = build_prompt(item)
    token_limit = MAX_NEW_TOKENS
    t0 = time.time()
    if debug:
        print("\n\n------------------\nInput is: ", item)
    out = llm(prompt, max_tokens = token_limit, temperature=TEMPERATURE, top_k=0, top_p=1.0, stop=["Input:"], echo=False)
    latency_ms = (time.time() - t0) * 1000.0
    text = out["choices"][0]["text"].strip()

    ## Testing the response, if json is incomplete retry once with higher max_tokens.
    t = normalize_text(text)
    obj, err = extract_single_json_balanced(t)
    if debug:
        print("\nJSON Error: ", err)
    if err == "unbalanced_braces":
        token_limit += 1000
        out = llm(prompt, max_tokens = token_limit, temperature=TEMPERATURE, top_k=0, top_p=1.0, stop=["Input:"], echo=False)
        latency_ms = (time.time() - t0) * 1000.0
        text = out["choices"][0]["text"].strip()

    ## Passed the text to a func which cleans and extract the 1st json object even if we have multiple jsons. 
    reasons = []
    if debug:
        print("\nRAW TEXT is: ", text)
    gen, rfix = auto_repair_once(text)
    reasons += rfix

    if debug:
        print("\nResponse is: ", gen)
    ok_req, r_req = v_required(gen, REQUIRED_KEYS)
    # ok_opt, r_opt = v_options_order(gen, src_opts)

    # if debug:
    ok = (rfix == "" or rfix == ["non_question"]) and ok_req

    if not ok:
        if r_req: reasons.append(r_req)

    tokens_out = out.get("usage", {}).get("completion_tokens", 0)
    
    resp = dict(
        raw_input = item.get("text",""),
        valid=int(ok),
        reason= "|".join(reasons) if reasons else "",
        latency_ms=round(latency_ms,2),
        max_token_limit = int(token_limit),
        used_tokens_out=int(tokens_out),
        json_obj=gen,
        response=text
    )
    return resp



# -------------- UI

st.title("Survey Automation (Local CPU)")
pwd = st.text_input("Password", type="password")
if pwd != APP_PASSWORD:
    st.stop()

st.caption(f"Run config: model={MODEL_PATH} | temperature={TEMPERATURE} | default_max_new_tokens={MAX_NEW_TOKENS}")

uploaded = st.file_uploader("Upload questionnaire (.docx)", type=["docx"])
use_demo = st.checkbox("Use built-in demo items (no upload)")

if use_demo:
    items = [
        {"text":"\nS2) STATE\nIn what state do you live?\n"},
        {"text":"\nBalance to census / Create variable xRegion single punch:\nNortheast \nMidwest \nWest \nSouth \n"},
        {"text":"S3) GENDER\nWhat is your current gender identity? \nMale\nFemale\nGender fluid or non-binary\nPrefer not to answer\n"},
        {"text":"\nS4) AGE\nWhat is your age?\nAges 14-77\n"},
        {"text":"\nCreate xAge single punch variables: \nGenZ (14-26)\nMillennials (27-42)\nGenX (43-58)\nBoomers (59-77)\n"},
        {"text":"\nD4) KIDS \nAre there any children under age 18 living in your household?\nYes\nNo\n"},
        {"text":"\nAsk D4A if selected “Yes” in D4\nD4A) KIDS AGE\nAre there any children under age 18 living in your household?\nYes\nNo\n"},
        {"text":"\nAdditional Demo Information \nD1) MARITAL STATUS\nWhat is your marital status?\nSingle, never married\nLiving with partner\nMarried\nSeparated/Divorced/Widowed\n"},
        {"text":"Show Q4 only for those evaluating the name ‘TEEM’\nTEEM PURCHASE IMPACT\n“Teem” was the name of a lemon-lime soft drink available in the 1960’s through the 1980’s. How does this knowledge affect your interest in purchasing a new lemon-lime soft drink called Teem?\nMuch more likely to purchase\nSomewhat more likely to purchase\nNo impact\nSomewhat less likely to purchase\nMuch less likely to purchase\n"},
        {"text":"\nScreener–Only respondents who qualify on appropriate criteria proceed to main survey\n"}
    ]
elif uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded.getvalue())
        tmp_file_path = tmp_file.name

    items = parse_qnr(tmp_file_path)
    os.remove(tmp_file_path)
else:
    items = []

with st.expander("Help (on-call)"):
    st.markdown(
        "- Truncation: If json truncation found, app will automatically increase the max_new_token and retry once.\n"
        "- JSON errors: Further errors wont be processed and intimated as valid%=0\n"
        "- Non-questions: Will be {}\n"
        "- Valid% low: Try the calculation after excluding Non questions. As those will be displayed as {} and counted as valid%=0 which affect the overall average\n"
        "- Kept do_sample False, temperature 0.0 to avoid stochastic randomness and stable latency\n"
        "- Export CSV will give all the inputs, output and necessary information for checking.\n"
    )

if items:
    if st.button("Run extraction"):
        st.write("Start: ", datetime.now())
        
        if use_demo:
            llm = load_llm(2048)
        else:
            llm = load_llm(3500)
        rows = []
        for item in items:
            rows.append(run_item(llm, item))
        st.write("End: ", datetime.now())
        df = pd.DataFrame(rows)

        st.dataframe(df[["raw_input", "valid", "latency_ms", "max_token_limit", "used_tokens_out", "reason", "json_obj"]].astype(str))

        st.subheader("Run report")

        vp = 100.0 * (pd.DataFrame(rows)["valid"].mean()) if rows else 0.0
        if vp < VALID_WARN_THRESHOLD:
            st.warning(f"Valid% {vp:.1f} below {VALID_WARN_THRESHOLD:.1f}")

        ## Download full export
        export = [dict(Raw_Input=r['raw_input'], Valid_JSON=r['valid'], LLM_Response=r['json_obj']) for r in rows]
        st.download_button("Export data.csv", pd.DataFrame(export).to_csv(index=False), file_name="export.csv",on_click="ignore")
