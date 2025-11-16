import logging
import os, sys
import time
import base64
import argparse
import json
from pathlib import Path
# working directory = /source/autonn_core/tangochat
COMMON_ROOT     = Path("/shared/common")
DATASET_ROOT    = Path("/shared/datasets")
MODEL_ROOT      = Path("/shared/models")

CORE_DIR        = Path(__file__).resolve().parent.parent # /source/autonn_core
sys.path.append(str(CORE_DIR))
CFG_DIR         = CORE_DIR / 'tangochat' / 'common' / 'cfg'
# HF_HOME         = os.environ.get('HF_HOME', '/root/.cache/huggingface')

import torch
import streamlit as st
# import huggingface_hub
# from tangochat.loader.download import (  
#     download_model,
#     list_model,
#     remove_model,
#     _get_diretory_size
# )
from tangochat.inference.generate import (  
    Generator,
    BuilderArgs,
    TokenizerArgs,
    GeneratorArgs
)
from tangochat.tuner.rag import (
    load_and_retrieve_docs,
    load_and_retrieve_docs_with_gpt,
    get_rag_formatted_prompt,
)
import ollama

# logging ----------------------------------------------------------------------
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# def run_tangochat():
# page config ------------------------------------------------------------------
st.set_page_config(
    page_title="TangoChat - LLMOps Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "TangoChat - LLMOps Platform by ETRI & Prosolution"
    }
)

# Custom CSS for enhanced dark theme with blue accents
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #0E1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1A1F2E;
    }

    /* Card-style containers */
    .model-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3D 100%);
        border: 1px solid #1E88E5;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(30, 136, 229, 0.1);
        transition: all 0.3s ease;
    }

    .model-card:hover {
        border-color: #42A5F5;
        box-shadow: 0 6px 12px rgba(30, 136, 229, 0.2);
        transform: translateY(-2px);
    }

    /* Status indicators */
    .status-active {
        color: #4CAF50;
        font-weight: 600;
    }

    .status-inactive {
        color: #757575;
    }

    /* Title styling */
    .main-title {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1E88E5 0%, #42A5F5 50%, #64B5F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Section headers */
    .section-header {
        color: #1E88E5;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1E88E5;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #1E88E5 0%, #1976D2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-weight: 600;
        transition: all 0.3s ease;
        white-space: nowrap;
        min-height: 2.5rem;
        font-size: 0.875rem;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #1976D2 0%, #1565C0 100%);
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
    }

    /* Sidebar button adjustments */
    [data-testid="stSidebar"] .stButton>button {
        padding: 0.3rem 0.5rem;
        font-size: 0.75rem;
        min-height: 1.8rem;
        width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Sidebar column spacing */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }

    /* Progress indicators */
    .stProgress > div > div {
        background-color: #1E88E5;
    }

    /* Chat message styling */
    .stChatMessage {
        background-color: #1A1F2E;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* User message - right aligned with icon on right */
    [data-testid="stChatMessageContent"]:has(+ [data-testid="chatAvatarIcon-user"]) {
        margin-left: auto;
        margin-right: 0;
    }

    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        margin-left: 20%;
        margin-right: 0;
    }

    /* Assistant message - left aligned with icon on left */
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        margin-left: 0;
        margin-right: 20%;
    }

    /* System message styling */
    .stChatMessage:has([data-testid="chatAvatarIcon-system"]) {
        background: #252B3D;
        border-left: 4px solid #1E88E5;
    }

    /* Model info display */
    .model-info {
        background: #1A1F2E;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }

    /* Radio button styling */
    .stRadio > label {
        color: #FAFAFA;
        font-weight: 500;
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #1A1F2E;
        color: #FAFAFA;
        border: 1px solid #1E88E5;
        border-radius: 8px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1A1F2E;
        border-radius: 8px;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# init -------------------------------------------------------------------------
st.session_state.uploader_key = 0

def reset_per_message_state():
    _update_uploader_key()

def _update_uploader_key():
    st.session_state.uploader_key = int(time.time())

start_state = [
    {
        "role": "system",
        "content": "**Configure your AI model in the sidebar** → Download → Apply → Chat",
    },
    {
        "role": "assistant",
        "content": "Welcome to TangoChat! I'm ready to assist you. Please select a model from the sidebar to get started."
    },
]

if "messages" not in st.session_state:
    st.session_state['messages'] = start_state

if "token" not in st.session_state:
    st.session_state.token = {'hf_token': ""}

if "brain" not in st.session_state:
    args = argparse.Namespace()
    args_path = CFG_DIR / 'args_chat.json'
    if os.path.isfile(args_path):
        with open(args_path, 'r') as f:
            args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
    st.session_state.brain = {
        "local_gen_name": None,
        "local_gen_obj": None,
        "local_gen_args": args
    }

if "bye" not in st.session_state:
    st.session_state.bye = False

if "img_prompt" not in st.session_state:
    st.session_state.img_prompt = "invisible"

if "rag" not in st.session_state:
    st.session_state.rag = {"active": False,
                            "url": "",
                            "embed": "",
                            "retreiver": None,}

# huggingface ------------------------------------------------------------------
# hf_token_cache = f'{HF_HOME}/token'
# if os.path.isfile(hf_token_cache):
#     with open(hf_token_cache, 'r') as f:
#         cached_token = f.read()
#     st.session_state.token = {"hf_token": cached_token}

# @st.dialog("Hugging Face Token")
# def get_token_and_login():
#     # with st.form(key='hf_token_submit_form'):
#     "If you do not have any, " + \
#     "**[get a new huggingface token](https://huggingface.co/settings/tokens)**"
#     hf_token = st.text_input(
#         label = "Access Token",
#         key = "huggingface_token",
#         type = 'password',
#         value = st.session_state.token['hf_token'],
#         help ='Access tokens authenticate your identity to the Hugging Face Hub ' + \
#             'and allow Tango+Chat to download LLMs based on token permissions.')
#     if st.button("Submit"):
#         st.session_state.token['hf_token'] = hf_token
#         success = login_hf()
#         if success:
#             st.rerun()

# def login_hf():
#     hf_token = st.session_state.token['hf_token']
#     try:
#         logger.info(f"Logging in the Hugging Face Hub with token: {hf_token}")
#         huggingface_hub.login(token=hf_token, write_permission=True, add_to_git_credential=True)
#         logger.info(f"Success logging in")
#         success = True
#     except Exception as e:
#         st.warning(f"***{e} Please input a valid token.***")
#         logger.warning(f"{e} Please input a valid token.")
#         st.session_state.token['hf_token'] = ""
#         success = False
#         logger.warning(f"Fail logging in")
#     finally:
#         return success

# exit -------------------------------------------------------------------------
if st.session_state.bye:
    logger.info("Completed\n")
    st.stop()


def switch_ollama_model():
    model_name = st.session_state.brain['local_gen_name']
    # logger.info(f"{st.session_state.brain}")
    if model_name is None:
        return
    st.session_state.brain['local_gen_obj'] = model_name
    # if st.session_state.rag['active']:
    #     st.session_state.rag = {"active": False,
    #                     "url": None,
    #                     "embed": None,
    #                     "retreiver": None,}
    st.balloons()
    
    st.session_state['messages'] = start_state
    for msg in st.session_state.messages:
        if msg['role'] == 'system':
            m_name = lists_for_ollama[model_name]
            msg['content'] = f"**{m_name}** is now active and ready to chat!"
            if 'LLaVA' in m_name:
                st.session_state.img_prompt = 'visible'

# Sidebar menu ----------------------------------------------------------------
lists_for_ollama = {
            "llama3.2"      : "Llama 3.2 (3B)",
            "llama3.1"      : "Llama 3.1 (8B)",
            "phi3.5"        : "Phi 3.5 (4B)",
            "mistral"       : "Mistral 0.3 (7B)",
            "neural-chat"   : "Neural-Chat (7B)",
            "codellama"     : "CodeLlama (7B)",
            "llava"         : "LLaVA 1.6 (8B)",
            "gemma2"        : "Gemma 2 (9B)",
            "qwen2.5"       : "Qwen 2.5 (7B)",
            "bnksys/yanolja-eeve-korean-instruct-10.8b": "EEVE Korean (11B)",
        }

model_providers = {
    "llama3.2": "Meta",
    "llama3.1": "Meta",
    "phi3.5": "Microsoft",
    "mistral": "Mistral AI",
    "neural-chat": "Intel",
    "codellama": "Meta",
    "llava": "UW-Madison",
    "gemma2": "Google",
    "qwen2.5": "Alibaba",
    "bnksys/yanolja-eeve-korean-instruct-10.8b": "Yanolja",
}

embed_lists = {
    "mxbai-embed-large": "MXBai-embed-large (334M)",
    "nomic-embed-text": "Nomic-embed-text (137M)",
    "all-minilm": "All-miniLM (23M)",
}

embed_providers = {
    "mxbai-embed-large": "MixedBread AI",
    "nomic-embed-text": "Nomic AI",
    "all-minilm": "SBERT.net",
}

with st.sidebar:
    st.markdown('<p class="section-header">MODEL MANAGEMENT</p>', unsafe_allow_html=True)

    # Get available models
    dn_lists = ollama.list()['models']
    local_lists = []
    local_model_keys = []
    for m in dn_lists:
        key = f"{m['model'].split(':')[0]}"
        value = lists_for_ollama.get(key, None)
        if value is not None:
            local_lists.append(value)
            local_model_keys.append(key)

    # Download Section
    with st.expander("📥 Download Models", expanded=False):
        st.markdown("**Select a model to download:**")

        # Create columns for model cards
        for model_key, model_name in lists_for_ollama.items():
            provider = model_providers.get(model_key, "Unknown")
            is_downloaded = model_key in local_model_keys

            col1, col2 = st.columns([2.5, 1.2])
            with col1:
                st.markdown(f"**{model_name}**")
                st.caption(f"Provider: {provider}")
            with col2:
                if is_downloaded:
                    st.markdown("✅ Downloaded")
                else:
                    if st.button("⬇ DL", key=f"dl_{model_key}"):
                        with st.status("Downloading...", expanded=True) as sts:
                            start = time.time()
                            ollama.pull(model_key)
                            elapsed_time = time.time() - start
                            st.write(f"✓ Completed in {elapsed_time:.2f}s")
                        sts.update(label=f"✓ {model_name} downloaded", state="complete")
                        st.rerun()
            st.divider()

    # Delete Section
    with st.expander("🗑️ Delete Models", expanded=False):
        if len(local_lists) > 0:
            st.markdown("**Remove models to free up storage:**")
            for idx, (model_name, model_key) in enumerate(zip(local_lists, local_model_keys)):
                col1, col2 = st.columns([2.5, 1.2])
                with col1:
                    st.markdown(f"**{model_name}**")
                with col2:
                    if st.button("🗑 Del", key=f"del_{model_key}"):
                        logger.info(f"Deleting model: {model_name}")
                        ollama.delete(model_key)
                        st.rerun()
                if idx < len(local_lists) - 1:
                    st.divider()
        else:
            st.info("No models installed")

    st.markdown("---")

    # RUN Section
    st.markdown('<p class="section-header">RUN MODEL</p>', unsafe_allow_html=True)

    if len(local_lists) > 0:
        st.markdown("**Select active model:**")

        current_model = st.session_state.brain.get('local_gen_name', None)
        current_display = lists_for_ollama.get(current_model, None) if current_model else None

        selected_idx = local_lists.index(current_display) if current_display in local_lists else 0

        selected_model_name = st.selectbox(
            "Active Model",
            local_lists,
            index=selected_idx,
            label_visibility="collapsed",
            key="model_selector"
        )

        # Find the key for selected model
        selected_model_key = None
        for k, v in lists_for_ollama.items():
            if v == selected_model_name:
                selected_model_key = k
                break

        if st.button("🚀 Apply Model", use_container_width=True):
            logger.info(f"Applying model: {selected_model_name}")
            st.session_state.brain['local_gen_name'] = selected_model_key
            st.session_state.brain['local_gen_obj'] = None
            st.session_state.rag = {
                "active": False,
                "url": None,
                "embed": None,
                "retreiver": None,
            }
            st.rerun()

        # Display current model status
        if current_model:
            st.markdown(f"""
            <div class="model-info">
                <strong>Active Model:</strong><br>
                <span class="status-active">● {lists_for_ollama[current_model]}</span><br>
                <small style="color: #757575;">Provider: {model_providers.get(current_model, 'Unknown')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No models available. Please download a model first.")

    st.markdown("---")

    # RAG Section
    st.markdown('<p class="section-header">RAG CONFIGURATION</p>', unsafe_allow_html=True)

    with st.expander("🔧 Embedding Model", expanded=False):
        st.markdown("**Select embedding model:**")
        for emb_key, emb_name in embed_lists.items():
            provider = embed_providers.get(emb_key, "Unknown")
            col1, col2 = st.columns([2.5, 1.2])
            with col1:
                st.markdown(f"**{emb_name}**")
                st.caption(f"Provider: {provider}")
            with col2:
                if st.button("⬇ Pull", key=f"emb_{emb_key}"):
                    with st.status("Downloading...", expanded=True) as sts1:
                        start = time.time()
                        ollama.pull(emb_key)
                        elapsed_time = time.time() - start
                        st.write(f"✓ Completed in {elapsed_time:.2f}s")
                    st.session_state.rag['active'] = True
                    st.session_state.rag['embed'] = emb_key
                    sts1.update(label=f"✓ {emb_name} ready", state="complete")
                    st.rerun()
            st.divider()

        if st.session_state.rag['embed']:
            st.success(f"✓ Using: {embed_lists[st.session_state.rag['embed']]}")

    st.markdown("**Document Source:**")
    _url = st.text_input(
        "URL",
        placeholder="e.g. https://github.com/ML-TANGO/TANGO",
        label_visibility="collapsed",
        key="rag_url_input"
    )
    st.session_state.rag['url'] = _url

    if st.button("🔍 Retrieve Documents", use_container_width=True):
        if st.session_state.rag.get('embed') and _url:
            logger.info(f'Starting RAG retrieval from: {_url}')
            with st.status("Retrieving documents...", expanded=False) as sts2:
                start = time.time()
                import chromadb
                _emb_model = st.session_state.rag['embed']
                logger.info(f'Embedding model: {_emb_model}')
                logger.info(f'URL: {_url}')
                _retriever = load_and_retrieve_docs_with_gpt(_url, _emb_model)
                elapsed_time = time.time() - start
                st.write(f"✓ Completed in {elapsed_time:.2f}s")
                st.session_state.rag['retriever'] = _retriever
                st.session_state['messages'] = start_state
                st.session_state.rag['active'] = True
                switch_ollama_model()
            sts2.update(label="✓ Documents retrieved", state="complete")
        else:
            if not st.session_state.rag.get('embed'):
                st.warning("⚠️ Please select an embedding model first")
            if not _url:
                st.warning("⚠️ Please enter a URL")

    # RAG Status Display
    if st.session_state.rag['active'] and _url:
        st.markdown(f"""
        <div class="model-info">
            <strong>RAG Status:</strong><br>
            <span class="status-active">● Active</span><br>
            <small style="color: #757575;">Source: {_url[:40]}...</small>
        </div>
        """, unsafe_allow_html=True)

# apply a local model to TangoChat ---------------------------------------------
def set_generator():
    model_name = st.session_state.brain['local_gen_name']
    args = st.session_state.brain['local_gen_args']
    logger.info(f"{st.session_state.brain}")
    if model_name is None:
        return
    args.model = model_name
    # args.prompt = prompt
    with st.status(label=f"Loading {model_name.upper()}... ", expanded=True) as l_sts:
        start = time.time()
        builder_args = BuilderArgs.from_args(args)
        speculative_builder_args = BuilderArgs.from_speculative_args(args)
        tokenizer_args = TokenizerArgs.from_args(args)
        generator_args = GeneratorArgs.from_args(args)
        TANGOCHAT = Generator(
            builder_args,
            speculative_builder_args,
            tokenizer_args,
            generator_args,
            args.profile,
            args.quantize,
            args.draft_quantize,
        )
        st.balloons()
        l_sts.update(
            label=f"Done. ({time.time() - start:.2f} sec)",
            state="complete",
        )
    st.session_state.brain['local_gen_obj'] = TANGOCHAT
    st.session_state.brain['local_gen_args'] = args
    st.session_state['messages'] = start_state
    for msg in st.session_state.messages:
        if msg['role'] == 'system':
            m_name = lists_for_ollama.get(model_name, model_name)
            msg['content'] = f"**{m_name}** is now active and ready to chat!"
    return





if st.session_state.brain['local_gen_obj'] == None:
    # set_generator()
    switch_ollama_model()

def show_img_loader():
    with st.sidebar:
        image_prompts = st.file_uploader(
            "Image Prompts",
            type=["jpeg"],
            accept_multiple_files=True,
            key=st.session_state.uploader_key,
        )

        for image in image_prompts:
            st.image(image)

if st.session_state.img_prompt == 'visible':
    show_img_loader()

# title ------------------------------------------------------------------------
st.markdown('<h1 class="main-title">TangoChat</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #757575; margin-top: -1.5rem; margin-bottom: 2rem;">LLMOps Platform powered by ETRI & Prosolution</p>', unsafe_allow_html=True)

st.divider()

if st.session_state.rag == 'active':
    with st.sidebar:
        url_prompt = st.chat_input(
            placeholder = "Let me know a specific URL to retreive",
            key = "url_prompt",
            )
        st.session_state.rag['url'] = url_prompt

# parsing messages -------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if type(msg['content']) is list:
            for content in msg['content']:
                st.write(content)
        elif type(msg['content']) is dict:
            st.write(msg['content'])
        elif type(msg['content']) is str:
            # Check if this is a user message with RAG context
            if msg['role'] == 'user' and "Context:" in msg['content'] and "Question:" in msg['content']:
                parts = msg['content'].split("Question:")
                if len(parts) > 1:
                    context_part = parts[0].replace("Context:", "").strip()
                    question_part = parts[1].strip()
                    st.write(question_part)
                    with st.expander("📄 View RAG Context", expanded=False):
                        st.markdown(context_part)
                else:
                    st.write(msg['content'])
            else:
                st.write(msg['content'])
        else:
            st.write(f"Unhandled content type: {type(msg['content'])}")

# user message input -----------------------------------------------------------
if prompt := st.chat_input():
    original_question = prompt
    rag_context = None

    if st.session_state.rag['active']:
        _retriever = st.session_state.rag.get('retriever', None)
        if _retriever is not None:
            formatted_prompt = get_rag_formatted_prompt(_retriever, prompt)
            # Extract context from formatted prompt
            if "Context:" in formatted_prompt and "Question:" in formatted_prompt:
                parts = formatted_prompt.split("Question:")
                if len(parts) > 1:
                    context_part = parts[0].replace("Context:", "").strip()
                    rag_context = context_part
            prompt = formatted_prompt
        else:
            st.warning("RAG가 활성화되어 있지만 retriever가 없습니다. 먼저 Retrieve를 눌러주세요.")

    user_message = {
        "role": "user",
        "content": prompt
    }

    if user_message["content"].lower() == 'bye':
        logger.info("Completed\n")
        st.session_state.bye == True
        st.rerun()

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(original_question)
        if rag_context:
            with st.expander("📄 View RAG Context", expanded=False):
                st.markdown(rag_context)

    # image_prompts = None
    reset_per_message_state()



    # completion generator -----------------------------------------------------
    with st.chat_message("assistant"), st.status(
        "Generating... ", expanded=True
    ) as status:
        # use api from other frameworks ----------------------------------------
        def get_streamed_completion(completion_generator):
            start = time.time()
            tokcount = 0
            for chunk in completion_generator:
                tokcount += 1
                # yield chunk.choices[0].delta.content  # open-ai style
                if chunk['done']:
                    break
                yield chunk['message']['content']       # ollama style
            
            speed = tokcount / (time.time() - start)
            status.update(
                label=f"Done, averaged {speed:.2f} tokens/second",
                state="complete",
            )

        # dumb brain -----------------------------------------------------------
        _LOREM_IPSUM = """
        Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, 
        sed do eiusmod tempor ncididunt ut labore et dolore magna aliqua. 
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi 
        ut aliquip ex ea commodo consequat.
        """
        _HANGUL_TEXT = """
        이 메세지는 아무런 의미가 없습니다. 
        화면을 위로 올린 후, **DOWNLOAD** 탭에서 모델을 다운로드 하세요. 
        **RUN** 탭에서 원하는 모델을 선택하고, **APPLY** 버튼을 눌러서 TangoChat을 깨우세요😀. 
        TangoChat을 종료하시려면 ***BYE*** 라고 쓰세요.
        """
        def temp_local_stream_data():
            start = time.time()
            for w in _LOREM_IPSUM.split(" "):
                yield w + " "
                time.sleep(0.02)
            
            import pandas as pd
            import numpy as np
            yield pd.DataFrame(
                np.random.randn(5,9),
                columns=["j", "k", "l", "m", "n", "o", "p", "q", "r"]
            )

            for w in _HANGUL_TEXT.split(" "):
                yield w + " "
                time.sleep(0.02)

            # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
            # st.bar_chart(chart_data)

            status.update(
                label=f"Done, elapsed time {time.time()-start:.2f} sec",
                state="complete",
            )

        # use local sources ----------------------------------------------------
        def get_answer():
            TANGOCHAT = st.session_state.brain['local_gen_obj']
            args = st.session_state.brain['local_gen_args']
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.time()
            tokcount = 0
            generator_args = GeneratorArgs.from_args(args)
            generator_args.prompt = prompt
            for chunk in TANGOCHAT.chat(generator_args):
                logger.info(f"TANGOCHAT.chat().type = {type(chunk)}")
                logger.info(chunk)
                tokcount += 1
                yield chunk[0]
            status.update(
                label="Done, averaged {:.2f} tockens/second".format(
                    tokcount / {time.time() - start}
                ),
                state="complete",
            )

        try:
            # response = st.write_stream(
            #     get_streamed_completion(
            #         client.chat.completions.create(
            #             model="llama3",
            #             messages=st.session_state.messages,
            #             max_tokens=response_max_tokens,
            #             temperature=temperature,
            #             stream=True,
            #         )
            #     )
            # )[0]  # open-ai style
            if st.session_state.brain['local_gen_obj'] is not None:
                logger.info(f"Local brain = {st.session_state.brain['local_gen_name']}")
                response = st.write_stream(
                    get_streamed_completion(
                        ollama.chat(
                            model=st.session_state.brain['local_gen_name'],
                            messages=st.session_state.messages,
                            # format='json',
                            stream=True,
                        )
                    )
                )   # ollama style
            else:
                logger.info("Local brain is not loaded...")
                response = st.write_stream(
                    temp_local_stream_data()
                )[0]    # dumb messages

        except Exception as e:
            response = st.error(f"Exception: {e}")
            logger.warning(f"Exception: {e}")
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )