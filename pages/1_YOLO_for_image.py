import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np

# Inicialize o objeto YOLO fora da função main
yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide',
                   page_icon='./images/object.png')

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Enviar Imagem')
    if image_file is not None:
        size_mb = image_file.size / (1024 ** 2)
        file_details = {"filename": image_file.name,
                        "filetype": image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        # validate file
        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('Tipo de arquivo imagem VALIDO (png ou jpeg)')
            return {"file": image_file,
                    "details": file_details}

        else:
            st.error('Tipo de arquivo de imagem INVALIDO')
            st.error('Upload only png, jpg, jpeg')
            return None

def main():
    st.header('Get Object Detection for any Image')
    st.write('Please Upload Image to get detections')
    
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object['file'])

        col1, col2 = st.columns(2)

        with col1:
            st.info('Pré-visualização da imagem')
            st.image(image_obj)

        with col2:
            st.subheader('Confira abaixo os detalhes do arquivo')
            st.json(object['details'])
            button = st.button('Descubra qual o Myxozoário pode estar presente em sua imagem')
            if button:
                with st.spinner("Obtendo Objets de imagem. Aguarde"):
                    # below command will convert
                    # obj to array
                    image_array = np.array(image_obj)
                    pred_img = yolo.predictions(image_array)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True

        if prediction:
            st.subheader("Imagem com a possivel detecção")
            st.caption("Detecção de Myxozoários")
            st.image(pred_img_obj)

if __name__ == "__main__":
    main()
