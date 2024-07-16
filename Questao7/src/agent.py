from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

logging.basicConfig(level=logging.INFO)

class PreparationTemplate:
    def __init__(self):
        self.system_template = """
            Você é um chefe de cozinha, com muitas habilidades gastronomicas internacionais, quando for perguntado sobre como preparar um determinado prato, deverá responder: 
             - Quais são os ingredientes necessários
             - Etapas para a preparação
             - Quais bebidas combinam com o prato
             - Informar qual sobremesa que combina com o prato
        """
        self.human_template = """
            ####{request}####
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template, input_variables=["request"])
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_prompt])
        
class Agent:
    def __init__(
        self,
        google_api_key,
        model="gemini-1.5-flash",
        temperature=0,
        verbose=True,
    ):
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        self.google_api_key = google_api_key
        self.model = model
        self.temperature = temperature
        self.verbose = verbose

        self.chat_model = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            google_api_key=self.google_api_key
        )

        
    def get_preparation(self, request):
        preparationTemplate = PreparationTemplate()
        
        preparation_agent = LLMChain(
            llm=self.chat_model,
            prompt=preparationTemplate.chat_prompt,
            verbose=self.verbose,
            output_key='agent_response'
        )

        overall_chain = SequentialChain(
            chains=[preparation_agent],
            input_variables=["request"],
            output_variables=["agent_response"],
            verbose=self.verbose
        )

        return overall_chain({"request": request}, return_only_outputs=True)