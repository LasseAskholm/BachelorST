# Importing packages: streamlit for the frontend, requests to make the api calls
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json

import GUI_API.Client as Client

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://lottie.host/88ab41fb-da1c-4fc6-a803-54966c5908e6/qLPlcizico.json")

class MakeCalls:
    def __init__(self, url: str = "http://localhost:8080/") -> None:
        """
        Constructor for the MakeCalls class. This class is used to perform API calls to the backend service.
        :param url: URL of the server. Default value is set to local host: http://localhost:8080
        """
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def model_list(self, service: str) -> dict:
        """
        Making an API request to backend service to get the details for each service. This function returns, list of names of trained models 
        :param service: NLP service that is being used.
        :return: List of names of trained models
        """
        model_info_url = self.url + f"api/v1/{service}/info"
        models = requests.get(url=model_info_url)
        return json.loads(models.text)

    def run_inference(
        self, service: str, model: str, text: str, query: str = None
    ) -> json:
        """
        This function is used to send the api request for the actual service for the specifed model to the
        :param service: String for the actual service.
        :param model: Model that is slected from the drop down.
        :param text: Input text that is used for analysis and to run inference.
        :param query: Input query for Information extraction use case.
        :return: results from the inference done by the model.
        """

        ## Post call to endpoints
        ## Might need modification where this is called


        # inference_enpoint = self.url + f"api/v1/{service}/predict"

        # payload = {"model": model.lower(), "text": text, "query": query.lower()}
        # result = requests.post(
        #     url=inference_enpoint, headers=self.headers, data=json.dumps(payload)
        # )

        if "bert" in model:
            #BERT call
            payload = {
                "inputs": "My name is Sarah Jessica Parker but you can call me Jessica",
            }

            result = Client.inference_bert(payload)

            return result
        

        elif "llama2" in model:
            ##Llama2 call

            payload = {"model": model.lower(), "text": text, "query": query.lower()}

            result = Client.inference_llama2_test(payload)

            print(result)
            return result['text']


class Display:
    def __init__(self):
        st.title("Name Entity Recognition Within the Defence Domain")
        st.sidebar.header("Select the NLP Service")
        self.service_options = st.sidebar.selectbox(
            label="",
            options=[
                "Homepage",
                "Named Entity Recognition",
            ],
        )
        self.service = {
            "Homepage": "about",
            "Named Entity Recognition": "ner",
        }

    def static_elements(self):
        return self.service[self.service_options]

    def dynamic_element(self, models_dict: dict):
        """
        This function is used to generate the page for each service.
        :param service: String of the service being selected from the side bar.
        :param models_dict: Dictionary of Model and its information. This is used to render elements of the page.
        :return: model, input_text run_button: Selected model from the drop down, input text by the user and run botton to kick off the process.
        """
        st.header(self.service_options)
        model_name = list()
        model_info = list()
        for i in models_dict.keys():
            model_name.append(models_dict[i]["name"])
            model_info.append(models_dict[i]["info"])
        st.sidebar.header("Model Information")
        for i in range(len(model_name)):
            st.sidebar.subheader(model_name[i])
            st.sidebar.info(model_info[i])
        model: str = st.selectbox("Select the Trained Model", model_name)
        input_text: str = st.text_area("Enter Text here")
        if self.service == "qna":
            query: str = st.text_input("Enter query here.")
        else:
            query: str = "None"
        run_button: bool = st.button("Run")
        return model, input_text, query, run_button


def main():
    st.set_page_config(layout="wide")
    page = Display()
    service = page.static_elements()
    apicall = MakeCalls()
    if service == "about":
        with st.container():
            st.write("---")
            left_column, right_column = st.columns(2)
            with left_column:
                st.write(
                    "This website allows for the interaction with the fine-tuned language models done as part of a comparative study highlighting the possabilities for using artifical intellegince to reduce resource cost and assist in the intellegence gathering used for decissionmaking within the defence domain."
                )
                st.write(
                    "To use this interface, select a base model from the dropdown in the side bar. From there you can further select which iteration of the model you wish to test. Each model has been trained on different amount of data and may output a more precise result depending of the iteration."
                )
                st.write(
                    "Fill in the text on which you want to run the service and the system will provide you with the models result."
                )
            with right_column:
                st_lottie(lottie_coding, height=300, key="coding")
    else:
        model_details = apicall.model_list(service=service)
        model, input_text, query, run_button = page.dynamic_element(model_details)
        if run_button:
            with st.spinner(text="Getting Results.."):
                result = apicall.run_inference(
                    service=service,
                    model=model.lower(),
                    text=input_text,
                    query=query.lower(),
                )
            st.write(result)


if __name__ == "__main__":
    main()
