import streamlit as st
from eval import generate, text

if __name__ == "__main__":
    st.set_page_config(
        page_title='Star★Beat★AI',
        page_icon='./MyGO_icon.svg.webp',
        layout='wide'
    )
    
    st.image("./MyGO_icon.svg.webp", width=75)
    st.title("Star★Beat★AI")
    
    length = st.slider(label="最大新token数", min_value=10, max_value=400, value=40)
    
    if st.button("确认"):
        st.subheader("模型回答：")
        cols = st.columns(3)
        res = generate(text, 50, 'file')
        for index in range(0, len(res), 3):
            cols = st.columns(3)
            for tag, item in enumerate(res[index:index + 3]):
                with cols[tag]:
                    with st.chat_message(name='user'):
                        st.markdown(item['prompt'])
                    with st.chat_message(name='assistant'):
                        st.markdown(item['ans'])
