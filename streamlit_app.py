import streamlit as st
import pandas as pd
import json
import re
from openai import OpenAI

st.set_page_config(page_title="Analisador de Risco de OS", page_icon="⚠️", layout="wide")

# ---------------- CONFIG ----------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o")

st.title("⚠️ Analisador de Padrões de Risco em Ordens de Serviço")
st.caption("Base histórica de reclamações reais → prompt calibrado automaticamente → score 0-100 de alerta")

# ---------------- SESSION ----------------
for k, v in {"df": None, "prompt_calibrado": None, "colunas": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- HELPERS ----------------
def carregar_base(arquivo):
    if arquivo.name.lower().endswith(".csv"):
        return pd.read_csv(arquivo, sep=None, engine="python")
    return pd.read_excel(arquivo)


def perfilar_base(df, max_cat=15, amostra=40):
    """Gera um resumo estatístico compacto da base para calibrar o prompt."""
    perfil = {"total_registros": int(len(df)), "colunas": {}}
    for col in df.columns:
        s = df[col]
        info = {"tipo": str(s.dtype), "nulos_pct": round(float(s.isna().mean() * 100), 1)}
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            info["estatisticas"] = {
                "min": float(d.get("min", 0)), "media": round(float(d.get("mean", 0)), 2),
                "mediana": float(s.median()) if s.notna().any() else None,
                "max": float(d.get("max", 0)),
            }
        else:
            vc = s.astype(str).value_counts().head(max_cat)
            info["cardinalidade"] = int(s.nunique())
            info["valores_frequentes"] = {str(k): int(v) for k, v in vc.items()}
        perfil["colunas"][col] = info

    amostras = df.sample(min(amostra, len(df)), random_state=42).astype(str).to_dict(orient="records")
    return perfil, amostras


def gerar_prompt_calibrado(perfil, amostras, contexto_usuario):
    meta = f"""Você é um engenheiro de prompts especialista em análise de risco operacional.

Recebeu o perfil de uma BASE HISTÓRICA DE RECLAMAÇÕES REAIS de Ordens de Serviço.
Sua tarefa: escrever um PROMPT DE SISTEMA definitivo que será usado para avaliar NOVAS OS
e retornar um score de 0 a 100 indicando probabilidade de a OS virar reclamação.

CONTEXTO DO NEGÓCIO FORNECIDO PELO USUÁRIO:
{contexto_usuario or "(não informado)"}

PERFIL ESTATÍSTICO DA BASE:
{json.dumps(perfil, ensure_ascii=False, indent=2)[:12000]}

AMOSTRAS REAIS DE RECLAMAÇÕES:
{json.dumps(amostras, ensure_ascii=False, indent=2)[:12000]}

O PROMPT que você vai escrever DEVE:
1. Explicitar os padrões de risco concretos identificados na base (combinações de campos,
   valores, faixas numéricas, categorias e textos que aparecem em reclamações).
2. Definir critérios de pontuação com pesos claros por padrão detectado.
3. Definir as faixas: 0-29 BAIXO, 30-59 MODERADO, 60-79 ALTO, 80-100 CRÍTICO.
4. Exigir saída ESTRITAMENTE em JSON:
{{"score": int, "classificacao": str, "padroes_detectados": [{{"padrao": str, "peso": int, "evidencia": str}}],
 "justificativa": str, "acoes_preventivas": [str]}}
5. Ser autossuficiente: quem ler o prompt não terá acesso à base original.

Responda APENAS com o texto do prompt de sistema, sem comentários ou markdown."""

    r = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role": "user", "content": meta}],
    )
    return r.choices[0].message.content.strip()


def analisar_os(prompt_sistema, dados_os):
    r = client.chat.completions.create(
        model=MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Avalie esta OS:\n{json.dumps(dados_os, ensure_ascii=False, indent=2)}"},
        ],
    )
    txt = re.sub(r"```json|```", "", r.choices[0].message.content).strip()
    return json.loads(txt)


def cor(score):
    return "#2E7D32" if score < 30 else "#F9A825" if score < 60 else "#EF6C00" if score < 80 else "#C62828"


# ---------------- ETAPA 1: BASE ----------------
st.header("1. Base histórica")
col1, col2 = st.columns([2, 1])

with col1:
    arquivo = st.file_uploader("Upload da base de reclamações (Excel/CSV)", type=["xlsx", "xls", "csv"])
with col2:
    st.info("Futuro: conexão direta SQL substituirá este upload.")

contexto = st.text_area(
    "Contexto do negócio (opcional, melhora a calibração)",
    placeholder="Ex: Ordens de serviço de reparo/troca de vidros automotivos, rede de afiliados...",
    height=80,
)

if arquivo:
    try:
        st.session_state.df = carregar_base(arquivo)
        st.session_state.colunas = list(st.session_state.df.columns)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

df = st.session_state.df

if df is not None:
    st.success(f"Base carregada: {len(df):,} registros × {len(df.columns)} colunas")
    with st.expander("Prévia da base"):
        st.dataframe(df.head(50), use_container_width=True)

    # ---------------- ETAPA 2: CALIBRAÇÃO ----------------
    st.header("2. Calibração do prompt")
    if st.button("🔧 Gerar prompt calibrado a partir da base", type="primary"):
        with st.spinner("Analisando padrões e gerando prompt..."):
            try:
                perfil, amostras = perfilar_base(df)
                st.session_state.prompt_calibrado = gerar_prompt_calibrado(perfil, amostras, contexto)
                st.success("Prompt calibrado gerado.")
            except Exception as e:
                st.error(f"Falha na calibração: {e}")

    if st.session_state.prompt_calibrado:
        with st.expander("Ver / editar prompt calibrado"):
            st.session_state.prompt_calibrado = st.text_area(
                "Prompt de sistema", st.session_state.prompt_calibrado, height=400, label_visibility="collapsed"
            )

        # ---------------- ETAPA 3: ANÁLISE ----------------
        st.header("3. Análise de OS")
        aba1, aba2 = st.tabs(["OS individual", "Lote (arquivo)"])

        with aba1:
            with st.form("form_os"):
                dados = {}
                cols = st.columns(3)
                for i, c in enumerate(st.session_state.colunas):
                    dados[c] = cols[i % 3].text_input(c, key=f"f_{c}")
                enviado = st.form_submit_button("Analisar risco", type="primary")

            if enviado:
                payload = {k: v for k, v in dados.items() if str(v).strip()}
                if not payload:
                    st.warning("Preencha ao menos um campo.")
                else:
                    with st.spinner("Avaliando..."):
                        try:
                            r = analisar_os(st.session_state.prompt_calibrado, payload)
                            s = int(r.get("score", 0))
                            st.markdown(
                                f"<div style='background:{cor(s)};padding:24px;border-radius:12px;text-align:center;color:#fff'>"
                                f"<div style='font-size:56px;font-weight:700'>{s}</div>"
                                f"<div style='font-size:20px'>{r.get('classificacao','')}</div></div>",
                                unsafe_allow_html=True,
                            )
                            st.progress(s / 100)
                            st.subheader("Justificativa")
                            st.write(r.get("justificativa", ""))
                            pd_ = r.get("padroes_detectados", [])
                            if pd_:
                                st.subheader("Padrões detectados")
                                st.dataframe(pd.DataFrame(pd_), use_container_width=True)
                            ac = r.get("acoes_preventivas", [])
                            if ac:
                                st.subheader("Ações preventivas")
                                for a in ac:
                                    st.markdown(f"- {a}")
                            with st.expander("JSON bruto"):
                                st.json(r)
                        except Exception as e:
                            st.error(f"Erro na análise: {e}")

        with aba2:
            lote = st.file_uploader("Arquivo com OS a avaliar", type=["xlsx", "xls", "csv"], key="lote")
            limite = st.number_input("Máximo de linhas", 1, 500, 20)
            if lote and st.button("Analisar lote", type="primary"):
                dfl = carregar_base(lote).head(int(limite))
                barra, res = st.progress(0.0), []
                for i, linha in enumerate(dfl.to_dict(orient="records")):
                    try:
                        r = analisar_os(st.session_state.prompt_calibrado, linha)
                        res.append({**linha, "score": r.get("score"), "classificacao": r.get("classificacao"),
                                    "justificativa": r.get("justificativa")})
                    except Exception as e:
                        res.append({**linha, "score": None, "classificacao": "ERRO", "justificativa": str(e)})
                    barra.progress((i + 1) / len(dfl))
                out = pd.DataFrame(res).sort_values("score", ascending=False, na_position="last")
                st.dataframe(out, use_container_width=True)
                st.download_button("⬇️ Baixar resultado (CSV)",
                                   out.to_csv(index=False).encode("utf-8-sig"),
                                   "analise_os.csv", "text/csv")
else:
    st.info("Faça o upload da base histórica para começar.")
