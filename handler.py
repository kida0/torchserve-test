import os
import json
import logging
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TestHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.device = None

    def initialize(self, context):
        """ initialize model, called during model laading time"""
        print('★★★★★ initialize ★★★★★')
        self.initialized = True
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Set device
        self.device = torch.device('cuda:' + str(properties.get('gpu_id')) if torch.cuda.is_available() else 'cpu')

        # Read model
        self.model = torch.jit.load(os.path.join(model_dir, 'test_model.pt'), map_location=self.device)
        self.model.eval()
        logger.debug("jit-model loaded successfully")

        # Read the mapping files
        try:
            with open(os.path.join(model_dir, "user2idx.json"), 'r') as json_file:
                self.user2idx = json.load(json_file)
            with open(os.path.join(model_dir, "question2idx.json"), 'r') as json_file:
                self.question2idx = json.load(json_file)
        except:
            logger.warning(
                "Missing the mapping files"
            )

    def preprocess(self, data):
        """ transform raw input into model input data """
        # unpacking
        inputs = data[0].get('data')
        if inputs is None:
            inputs = data[0].get('body')
        inputs = torch.tensor([list(map(lambda x: self.user2idx[x], inputs['userId'])),
                               list(map(lambda x: self.question2idx[x], inputs['questionId']))]).T
        return inputs

    def inference(self, inputs):
        """ internal inference methods """
        inputs = inputs.to(self.device)
        y_pred = self.model(inputs).squeeze().tolist()
        return [y_pred]

    def postprocess(self, inference_output):
        """ return inference result """
        print('Do nothing')
        return inference_output

    def handle(self, data, context):
        """ invoke by torchserve for prediction request """
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data
