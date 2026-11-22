import streamlit as st

from agent import MODEL_NAME, build_agent, chat_with_agent

st.set_page_config(page_title="Restaurant Assistant", page_icon="🍽️")
st.title("🍽️ Restaurant Assistant")
st.caption(f"Find restaurants, get details, and check reservation availability · {MODEL_NAME}")


@st.cache_resource
def get_agent():
    return build_agent()


try:
    agent = get_agent()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Try asking")
    st.markdown(
        "- Good Indian restaurants?\n- Tell me about Golden Wok\n"
        "- Is Spice Palace free for 4 on 2024-11-15?"
    )
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about restaurants or reservations..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, st.session_state.history = chat_with_agent(agent, prompt, st.session_state.history)
            except Exception as e:
                reply = f"Sorry, something went wrong: {e}"
        st.markdown(reply)
