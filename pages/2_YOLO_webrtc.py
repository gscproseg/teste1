import streamlit as st 
import asyncio
from streamlit_webrtc import (
    VideoProcessorBase,
    RTCConfiguration,
    webrtc_streamer
)
import av
from yolo_predictions import YOLO_Pred

# Carregar o modelo YOLO
yolocam = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

# Definir configuração RTC (WebRTC)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class YOLOVideoProcessor(VideoProcessorBase):
    async def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img_cam = frame.to_ndarray(format="bgr24")

            # Processamento assíncrono
            loop = asyncio.get_running_loop()
            pred_img_video = await loop.run_in_executor(None, yolocam.predictions, img_cam)

            # Adicionar mensagem de log para verificar as previsões
            st.write(f"Previsões: {pred_img_video}")

            return av.VideoFrame.from_ndarray(pred_img_video, format="bgr24")
        except Exception as e:
            st.error(f"Erro durante o processamento do frame: {e}")
            return frame

# Configurar e iniciar a transmissão WebRTC
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=YOLOVideoProcessor,
    rtc_configuration=rtc_configuration,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
)

    # Verificar o estado do WebRTC e adicionar logs
    st.write(f"Estado do WebRTC: {webrtc_ctx.state}")

if webrtc_ctx.state == "connected":
    st.write("Streaming de vídeo com detecção de objetos está ativo.")
else:
    st.write("Aguardando a transmissão de vídeo começar...")

# Exibir a interface do Streamlit
if webrtc_ctx.state == "connected":
    st.write("Streaming de vídeo com detecção de objetos está ativo.")
else:
    st.write("Aguardando a transmissão de vídeo começar...")
