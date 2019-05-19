from  chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

class SimpleChat():

    def __init__(self):
        self.chatbot = ChatBot('myBot',

                               logic_adapters=[
                                   {
                                       'import_path': 'chatterbot.logic.BestMatch'
                                   },
                                   {
                                       'import_path': 'chatterbot.logic.LowConfidenceAdapter',
                                       'threshold': 0.65,
                                       'default_response': 'NO'
                                   }
                               ],
                               )

        self.chatbot.set_trainer(ChatterBotCorpusTrainer)
        # self.chatbot.train("chatterbot.corpus.chinese.ai")

    def get_response(self, infos):
        # 返回信息
        return str(self.chatbot.get_response(infos))
