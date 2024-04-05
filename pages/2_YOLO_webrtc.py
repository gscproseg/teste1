import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from yolo_predictions import YOLO_Pred

# Carrega o modelo YOLO
yolo = YOLO_Pred('./models/best.onnx', './models/data.yaml')

def video_frame_callback(frame):
    try:
        img_vid = frame.to_ndarray(format="bgr24")
        
        # Realiza previsões com o modelo YOLO
        pred_img = yolo.predictions(img_vid)
        
        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")
    except Exception as e:
        st.error(f"Erro ao processar o frame: {e}")

# Inicia o aplicativo Streamlit com o fluxo de vídeo
try:
    webrtc_streamer(key="cam",
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False})
except Exception as e:
    st.error(f"Erro ao iniciar o stream: {e}")
