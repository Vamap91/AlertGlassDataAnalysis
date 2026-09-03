import streamlit as st
import pandas as pd
import numpy as np
import json, re, pickle, io, hashlib
from openai import OpenAI

st.set_page_config(page_title="Analisador de Risco de OS", page_icon="⚠️", layout="wide")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
CHAT_MODEL  = st.secrets.get("OPENAI_MODEL", "gpt-4o")
EMB_MODEL   = st.secrets.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH       = 96

st.title("⚠️ Analisador de Padrões de Risco em Ordens de Serviço")
st.caption("Base vetorizada (OpenAI Embeddings) → Camada 1: prompt padronizado → Camada 2: score 0-100 da OS")

for k, v in {"df": None, "colunas": [], "emb": None, "textos": [],
             "prompt_camada1": None, "assinatura": None}.items():
    st.session_state.setdefault(k, v)


# ============ HELPERS ============
def carregar(arquivo):
    if arquivo.name.lower().endswith(".csv"):
        return pd.read_csv(arquivo, sep=None, engine="python")
    return pd.read_excel(arquivo)


def texto_canonico(registro, colunas):
    """Mesma serialização para base histórica e para OS nova — garante espaço vetorial coerente."""
    partes = []
    for c in colunas:
        v = registro.get(c, "")
        if pd.isna(v) or str(v).strip() in ("", "nan", "None"):
            continue
        partes.append(f"{c}: {str(v).strip()}")
    return " | ".join(partes)


def embed(textos):
    vetores = []
    barra = st.progress(0.0, text="Vetorizando...")
    for i in range(0, len(textos), BATCH):
        lote = [t[:8000] if t.strip() else "vazio" for t in textos[i:i + BATCH]]
        r = client.embeddings.create(model=EMB_MODEL, input=lote)
        vetores.extend([d.embedding for d in r.data])
        barra.progress(min((i + BATCH) / len(textos), 1.0), text=f"Vetorizando... {min(i+BATCH,len(textos))}/{len(textos)}")
    barra.empty()
    M = np.array(vetores, dtype=np.float32)
    return M / np.linalg.norm(M, axis=1, keepdims=True)  # normalizado → cosseno = produto interno


def buscar_similares(vetor_os, k=8):
    sims = st.session_state.emb @ vetor_os
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


def perfilar(df, max_cat=15):
    perfil = {"total_registros": int(len(df)), "colunas": {}}
    for col in df.columns:
        s = df[col]
        info = {"tipo": str(s.dtype), "nulos_pct": round(float(s.isna().mean() * 100), 1)}
        if pd.api.types.is_numeric_dtype(s):
            info["estatisticas"] = {
                "min": float(s.min()), "media": round(float(s.mean()), 2),
                "mediana": float(s.median()), "p90": float(s.quantile(0.9)), "max": float(s.max()),
            }
        else:
            info["cardinalidade"] = int(s.nunique())
            info["valores_frequentes"] = {str(a): int(b) for a, b in s.astype(str).value_counts().head(max_cat).items()}
        perfil["colunas"][col] = info
    return perfil


def amostras_diversificadas(n=40):
    """Amostra espalhada no espaço vetorial (k-means leve) para a Camada 1 ver a variedade real de reclamações."""
    emb, textos = st.session_state.emb, st.session_state.textos
    n = min(n, len(textos))
    rng = np.random.default_rng(42)
    centros = [int(rng.integers(len(emb)))]
    dist = 1 - emb @ emb[centros[0]]
    while len(centros) < n:
        prox = int(np.argmax(dist))
        centros.append(prox)
        dist = np.minimum(dist, 1 - emb @ emb[prox])
    return [textos[i] for i in centros]


def gerar_camada1(perfil, amostras, contexto):
    meta = f"""Você é engenheiro de prompts especialista em risco operacional.
Recebeu o perfil de uma BASE HISTÓRICA DE RECLAMAÇÕES REAIS de Ordens de Serviço, já vetorizada.

Sua tarefa: escrever o PROMPT DE SISTEMA PADRONIZADO (Camada 1) que será usado para avaliar NOVAS OS.
Esse prompt receberá, além da OS nova, uma lista de RECLAMAÇÕES HISTÓRICAS SIMILARES recuperadas por
similaridade vetorial, com seus respectivos scores de similaridade (0 a 1).

CONTEXTO DO NEGÓCIO:
{contexto or "(não informado)"}

PERFIL ESTATÍSTICO DA BASE:
{json.dumps(perfil, ensure_ascii=False, indent=2)[:12000]}

AMOSTRAS DIVERSIFICADAS (cobrindo os agrupamentos do espaço vetorial):
{json.dumps(amostras, ensure_ascii=False, indent=2)[:14000]}

O PROMPT que você vai escrever DEVE:
1. Listar os padrões de risco concretos observados (combinações de campos, faixas numéricas,
   categorias, expressões textuais recorrentes nas reclamações).
2. Atribuir pesos numéricos a cada padrão.
3. Instruir a ponderar a evidência vetorial: quantos similares foram recuperados, quão altos os
   scores de similaridade e o quanto os padrões deles se repetem na OS avaliada.
4. Definir faixas: 0-29 BAIXO, 30-59 MODERADO, 60-79 ALTO, 80-100 CRÍTICO.
5. Exigir saída ESTRITAMENTE em JSON:
{{"score": int, "classificacao": str,
  "padroes_detectados": [{{"padrao": str, "peso": int, "evidencia": str}}],
  "similares_relevantes": [str], "justificativa": str, "acoes_preventivas": [str]}}
6. Ser autossuficiente — quem executar não terá acesso à base bruta.

Responda APENAS com o texto do prompt, sem markdown nem comentários."""
    r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.2,
                                       messages=[{"role": "user", "content": meta}])
    return r.choices[0].message.content.strip()


def analisar(registro_os, k=8):
    texto = texto_canonico(registro_os, st.session_state.colunas)
    vetor = np.array(client.embeddings.create(model=EMB_MODEL, input=[texto]).data[0].embedding, dtype=np.float32)
    vetor /= np.linalg.norm(vetor)
    viz = buscar_similares(vetor, k)
    bloco = "\n".join(f"[similaridade {s:.3f}] {st.session_state.textos[i]}" for i, s in viz)

    user = f"""OS A SER AVALIADA:
{texto}

RECLAMAÇÕES HISTÓRICAS SIMILARES (recuperadas por vetorização):
{bloco}

Retorne o JSON no formato exigido."""
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": st.session_state.prompt_camada1},
                  {"role": "user", "content": user}])
    saida = json.loads(re.sub(r"```json|```", "", r.choices[0].message.content).strip())
    return saida, viz


def cor(s):
    return "#2E7D32" if s < 30 else "#F9A825" if s < 60 else "#EF6C00" if s < 80 else "#C62828"


# ============ 1. BASE + VETORIZAÇÃO ============
st.header("1. Base histórica e vetorização")
c1, c2 = st.columns([2, 1])
with c1:
    arq = st.file_uploader("Base de reclamações (Excel/CSV)", type=["xlsx", "xls", "csv"])
with c2:
    idx_file = st.file_uploader("Índice já vetorizado (.pkl)", type=["pkl"])

contexto = st.text_area("Contexto do negócio (opcional)", height=70,
                        placeholder="Ex: OS de reparo/troca de vidros automotivos, rede de afiliados...")

if idx_file:
    pack = pickle.load(idx_file)
    st.session_state.update(emb=pack["emb"], textos=pack["textos"], colunas=pack["colunas"],
                            df=pack["df"], prompt_camada1=pack.get("prompt_camada1"),
                            assinatura=pack.get("assinatura"))
    st.success(f"Índice restaurado: {len(pack['textos']):,} vetores.")

if arq:
    try:
        st.session_state.df = carregar(arq)
        st.session_state.colunas = list(st.session_state.df.columns)
    except Exception as e:
        st.error(f"Erro ao ler: {e}")

df = st.session_state.df

if df is not None:
    st.success(f"Base: {len(df):,} registros × {len(df.columns)} colunas")
    with st.expander("Prévia"):
        st.dataframe(df.head(50), use_container_width=True)

    if st.button("🧬 Vetorizar base", type="primary"):
        registros = df.to_dict(orient="records")
        st.session_state.textos = [texto_canonico(r, st.session_state.colunas) for r in registros]
        try:
            st.session_state.emb = embed(st.session_state.textos)
            st.session_state.assinatura = hashlib.md5("".join(st.session_state.colunas).encode()).hexdigest()[:8]
            st.success(f"{len(st.session_state.textos):,} registros vetorizados ({st.session_state.emb.shape[1]} dim).")
        except Exception as e:
            st.error(f"Falha na vetorização: {e}")

if st.session_state.emb is not None:
    # ============ 2. CAMADA 1 ============
    st.header("2. Camada 1 — prompt padronizado")
    if st.button("🔧 Gerar prompt a partir da base vetorizada"):
        with st.spinner("Extraindo padrões e escrevendo o prompt..."):
            try:
                st.session_state.prompt_camada1 = gerar_camada1(
                    perfilar(st.session_state.df), amostras_diversificadas(40), contexto)
                st.success("Camada 1 gerada.")
            except Exception as e:
                st.error(f"Falha: {e}")

    if st.session_state.prompt_camada1:
        with st.expander("Ver / editar prompt da Camada 1"):
            st.session_state.prompt_camada1 = st.text_area(
                "p1", st.session_state.prompt_camada1, height=400, label_visibility="collapsed")

        buf = io.BytesIO()
        pickle.dump({"emb": st.session_state.emb, "textos": st.session_state.textos,
                     "colunas": st.session_state.colunas, "df": st.session_state.df,
                     "prompt_camada1": st.session_state.prompt_camada1,
                     "assinatura": st.session_state.assinatura}, buf)
        st.download_button("⬇️ Baixar índice + prompt (.pkl)", buf.getvalue(),
                           "indice_os.pkl", "application/octet-stream")

        # ============ 3. CAMADA 2 ============
        st.header("3. Camada 2 — análise da OS")
        k = st.slider("Similares recuperados por OS", 3, 20, 8)
        aba1, aba2 = st.tabs(["OS individual", "Lote"])

        with aba1:
            with st.form("os"):
                dados, cols = {}, st.columns(3)
                for i, c in enumerate(st.session_state.colunas):
                    dados[c] = cols[i % 3].text_input(c, key=f"f_{c}")
                ok = st.form_submit_button("Analisar risco", type="primary")

            if ok:
                payload = {k_: v for k_, v in dados.items() if str(v).strip()}
                if not payload:
                    st.warning("Preencha ao menos um campo.")
                else:
                    with st.spinner("Avaliando..."):
                        try:
                            r, viz = analisar(payload, k)
                            s = int(r.get("score", 0))
                            st.markdown(
                                f"<div style='background:{cor(s)};padding:24px;border-radius:12px;text-align:center;color:#fff'>"
                                f"<div style='font-size:56px;font-weight:700'>{s}</div>"
                                f"<div style='font-size:20px'>{r.get('classificacao','')}</div></div>",
                                unsafe_allow_html=True)
                            st.progress(s / 100)
                            st.subheader("Justificativa"); st.write(r.get("justificativa", ""))
                            if r.get("padroes_detectados"):
                                st.subheader("Padrões detectados")
                                st.dataframe(pd.DataFrame(r["padroes_detectados"]), use_container_width=True)
                            if r.get("acoes_preventivas"):
                                st.subheader("Ações preventivas")
                                for a in r["acoes_preventivas"]:
                                    st.markdown(f"- {a}")
                            with st.expander("Reclamações similares recuperadas"):
                                st.dataframe(pd.DataFrame(
                                    [{"similaridade": round(sim, 3), "registro": st.session_state.textos[i]}
                                     for i, sim in viz]), use_container_width=True)
                            with st.expander("JSON bruto"):
                                st.json(r)
                        except Exception as e:
                            st.error(f"Erro: {e}")

        with aba2:
            lote = st.file_uploader("OS a avaliar", type=["xlsx", "xls", "csv"], key="lote")
            limite = st.number_input("Máximo de linhas", 1, 500, 20)
            if lote and st.button("Analisar lote", type="primary"):
                dfl = carregar(lote).head(int(limite))
                barra, res = st.progress(0.0), []
                for i, linha in enumerate(dfl.to_dict(orient="records")):
                    try:
                        r, viz = analisar(linha, k)
                        res.append({**linha, "score": r.get("score"), "classificacao": r.get("classificacao"),
                                    "top_similaridade": round(viz[0][1], 3),
                                    "justificativa": r.get("justificativa")})
                    except Exception as e:
                        res.append({**linha, "score": None, "classificacao": "ERRO", "justificativa": str(e)})
                    barra.progress((i + 1) / len(dfl))
                out = pd.DataFrame(res).sort_values("score", ascending=False, na_position="last")
                st.dataframe(out, use_container_width=True)
                st.download_button("⬇️ Baixar CSV", out.to_csv(index=False).encode("utf-8-sig"),
                                   "analise_os.csv", "text/csv")
else:
    st.info("Carregue a base e vetorize, ou restaure um índice `.pkl` existente.")
