### src_fastapi:
sudo docker compose -f streamlit_gui/src_fastapi/docker-compose.yml up -d
echo "Docker fastapi UP"

### src_streamlit:
#streamlit run streamlit_gui/src_streamlit/NLPfiy.py
python3 -m streamlit run streamlit_gui/src_streamlit/NLPfiy.py 

bash gui_down.sh